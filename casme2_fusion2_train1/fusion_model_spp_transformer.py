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
            # Global average over the l x l grid to get (B, C)
            feat = pooled.mean(dim=(-1, -2))  # (B, C)
            token = self.level_proj(feat)  # (B, d_model)
            tokens.append(token)
        # Stack into sequence: (B, L, d_model)
        return torch.stack(tokens, dim=1)


class LateFusionResNetWithSPPTransformer(nn.Module):
    """
    Late Fusion ResNet model with SPP tokens per branch and a lightweight
    shared Transformer Encoder to model cross-branch and cross-scale dependencies.

    - Two branches (RGB & Dynamic Image) produce L tokens each (L = len(spp_levels))
    - Tokens are concatenated forming a sequence of length 2*L (+ optional CLS token)
    - Learnable branch and level embeddings are added for positional encoding
    - A small TransformerEncoder (1-2 layers, 4 heads) encodes the sequence
    - CLS token output is used as fused representation for final classification
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True,
                 d_model: int = 256, spp_levels=(1, 2, 4), n_heads: int = 4, n_layers: int = 2, ff_multiplier: int = 4,
                 use_cls_token: bool = True):
        super().__init__()
        self.d_model = d_model
        self.spp_levels = spp_levels
        self.use_cls_token = use_cls_token

        # Create backbones
        self.rgb_backbone = self._create_resnet_backbone(in_channels=3, pretrained=pretrained)
        self.dynamic_backbone = self._create_resnet_backbone(in_channels=3, pretrained=pretrained)

        # Branch token builders
        self.rgb_tokens = ResNetBranchTokens(self.rgb_backbone, d_model=d_model, spp_levels=spp_levels)
        self.dynamic_tokens = ResNetBranchTokens(self.dynamic_backbone, d_model=d_model, spp_levels=spp_levels)

        # Positional embeddings: branch ID and scale level
        self.branch_embed = nn.Embedding(2, d_model)  # 0: RGB, 1: Dynamic
        self.level_embed = nn.Embedding(len(spp_levels), d_model)  # level indices 0..L-1

        # Optional CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_multiplier * d_model,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pre_ln = nn.LayerNorm(d_model)
        self.post_ln = nn.LayerNorm(d_model)

        # Classification head on CLS (or pooled) token
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(d_model, num_classes)

    def _create_resnet_backbone(self, in_channels, pretrained=True):
        """Create a ResNet-18 backbone up to layer4, removing final fc."""
        if pretrained:
            model = torchvision_resnet18(weights='IMAGENET1K_V1')
            # Modify first conv if in_channels != 3
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
        """
        Add learnable branch and level embeddings to tokens.
        tokens: (B, L, d_model)
        branch_id: 0 or 1
        """
        B, L, D = tokens.shape
        branch_vec = self.branch_embed(torch.tensor(branch_id, device=tokens.device))  # (D,)
        level_ids = torch.arange(L, device=tokens.device)
        level_vecs = self.level_embed(level_ids)  # (L, D)
        # Broadcast and add
        tokens = tokens + branch_vec.view(1, 1, D) + level_vecs.view(1, L, D)
        return tokens

    def forward(self, rgb_input, dynamic_input):
        # Build tokens for each branch
        rgb_tokens = self.rgb_tokens(rgb_input)           # (B, L, D)
        dynamic_tokens = self.dynamic_tokens(dynamic_input)  # (B, L, D)

        # Add positional encodings
        rgb_tokens = self._add_positional_embeddings(rgb_tokens, branch_id=0)
        dynamic_tokens = self._add_positional_embeddings(dynamic_tokens, branch_id=1)

        # Concatenate tokens: (B, 2L, D)
        tokens = torch.cat([rgb_tokens, dynamic_tokens], dim=1)

        # Pre-norm
        tokens = self.pre_ln(tokens)

        # Optionally prepend CLS token
        if self.use_cls_token:
            B = tokens.size(0)
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, 1 + 2L, D)

        # Encode
        encoded = self.transformer(tokens)  # (B, S, D)
        encoded = self.post_ln(encoded)

        # Use CLS token if present, else mean-pool
        if self.use_cls_token:
            fused = encoded[:, 0, :]  # (B, D)
        else:
            fused = encoded.mean(dim=1)  # (B, D)

        # Classification
        fused = self.dropout(fused)
        output = self.fc(fused)
        return output


def create_fusion_model(num_classes=3, dropout_p=0.5, pretrained=True):
    """
    Factory function to create the SPP + Transformer late fusion model.
    """
    return LateFusionResNetWithSPPTransformer(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)