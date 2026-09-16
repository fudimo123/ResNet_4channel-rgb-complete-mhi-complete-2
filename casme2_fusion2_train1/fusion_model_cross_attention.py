import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as torchvision_resnet18


class ResNetBranchTokens(nn.Module):
    """
    Extract layer4 feature map from ResNet-18, then build multi-scale tokens
    using adaptive pooling at levels (e.g., 1, 2, 4). For each level l, we
    apply adaptive avg pooling to (l, l) and then average over spatial to get
    a C-dim feature, project to d_model as a token.
    """
    def __init__(self, backbone: nn.Module, d_model: int = 256, spp_levels=(1, 2, 4)):
        super().__init__()
        self.backbone = backbone
        self.spp_levels = spp_levels
        self.d_model = d_model
        # Project from channel dimension (512) to transformer embed dim
        self.level_proj = nn.Linear(512, d_model)

    def forward(self, x):
        # Run backbone up to layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # (B, 512, H, W)

        B, C, H, W = x.shape
        tokens = []
        for l in self.spp_levels:
            pooled = F.adaptive_avg_pool2d(x, (l, l))  # (B, C, l, l)
            feat = pooled.mean(dim=(-1, -2))  # (B, C)
            token = self.level_proj(feat)  # (B, d_model)
            tokens.append(token)
        # Stack into sequence: (B, L, d_model)
        return torch.stack(tokens, dim=1)


class CoAttentionFusionModel(nn.Module):
    """
    Cross-modal Co-Attention fusion:
    - Two branches (RGB & Dynamic Image) produce L tokens each via SPP levels
    - Use multi-head attention to compute cross-attention (RGB queries attend Dynamic K/V and vice versa)
    - Aggregate attended sequences into vectors, concatenate, and classify
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True,
                 d_model: int = 256, spp_levels=(1, 2, 4), n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.spp_levels = spp_levels

        # Create backbones
        self.rgb_backbone = self._create_resnet_backbone(in_channels=3, pretrained=pretrained)
        self.dynamic_backbone = self._create_resnet_backbone(in_channels=3, pretrained=pretrained)

        # Branch token builders
        self.rgb_tokens = ResNetBranchTokens(self.rgb_backbone, d_model=d_model, spp_levels=spp_levels)
        self.dynamic_tokens = ResNetBranchTokens(self.dynamic_backbone, d_model=d_model, spp_levels=spp_levels)

        # Positional embeddings: branch ID and scale level (help alignment across modalities & scales)
        self.branch_embed = nn.Embedding(2, d_model)  # 0: RGB, 1: Dynamic
        self.level_embed = nn.Embedding(len(spp_levels), d_model)  # level indices 0..L-1

        # Cross-attention modules (bidirectional)
        self.attn_rgb_to_dyn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.attn_dyn_to_rgb = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        # LayerNorm for stability
        self.ln_rgb = nn.LayerNorm(d_model)
        self.ln_dyn = nn.LayerNorm(d_model)

        # Fusion & classification head
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(2 * d_model, num_classes)

    def _create_resnet_backbone(self, in_channels, pretrained=True):
        """Create a ResNet-18 backbone up to layer4, removing final fc."""
        if pretrained:
            model = torchvision_resnet18(weights='IMAGENET1K_V1')
            if in_channels != 3:
                original_weights = model.conv1.weight.clone()
                new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    if in_channels == 1:
                        new_conv1.weight[:, 0, :, :] = torch.mean(original_weights, dim=1)
                    else:
                        new_conv1.weight[:, :3, :, :] = original_weights
                        for i in range(3, in_channels):
                            new_conv1.weight[:, i, :, :] = torch.mean(original_weights, dim=1)
                model.conv1 = new_conv1
            model.fc = nn.Identity()
        else:
            raise NotImplementedError("Non-pretrained ResNet is not implemented for fusion model")
        return model

    def _add_positional_embeddings(self, tokens, branch_id: int):
        """Add learnable branch and level embeddings to tokens (B, L, D)."""
        B, L, D = tokens.shape
        branch_vec = self.branch_embed(torch.tensor(branch_id, device=tokens.device))  # (D,)
        level_ids = torch.arange(L, device=tokens.device)
        level_vecs = self.level_embed(level_ids)  # (L, D)
        tokens = tokens + branch_vec.view(1, 1, D) + level_vecs.view(1, L, D)
        return tokens

    def forward(self, rgb_input, dynamic_input):
        # Build tokens for each branch
        rgb_tokens = self.rgb_tokens(rgb_input)             # (B, L, D)
        dynamic_tokens = self.dynamic_tokens(dynamic_input) # (B, L, D)

        # Add positional encodings
        rgb_tokens = self._add_positional_embeddings(rgb_tokens, branch_id=0)
        dynamic_tokens = self._add_positional_embeddings(dynamic_tokens, branch_id=1)

        # LayerNorm pre-attention
        rgb_tokens = self.ln_rgb(rgb_tokens)
        dynamic_tokens = self.ln_dyn(dynamic_tokens)

        # Bidirectional cross-attention
        # RGB queries attend dynamic K/V
        rgb_attended, _ = self.attn_rgb_to_dyn(query=rgb_tokens, key=dynamic_tokens, value=dynamic_tokens)
        # Dynamic queries attend RGB K/V
        dyn_attended, _ = self.attn_dyn_to_rgb(query=dynamic_tokens, key=rgb_tokens, value=rgb_tokens)

        # Aggregate sequences into single vectors
        rgb_vec = rgb_attended.mean(dim=1)  # (B, D)
        dyn_vec = dyn_attended.mean(dim=1)  # (B, D)

        # Concatenate and classify
        fused = torch.cat([rgb_vec, dyn_vec], dim=1)  # (B, 2D)
        fused = self.dropout(fused)
        output = self.fc(fused)
        return output


def create_fusion_model(num_classes=3, dropout_p=0.5, pretrained=True):
    """Factory for Co-Attention fusion model."""
    return CoAttentionFusionModel(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)