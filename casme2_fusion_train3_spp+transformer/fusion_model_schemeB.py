import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as torchvision_resnet18


class SpatialPyramidPoolingTokens(nn.Module):
    """
    SPP that returns per-level tokens preserving spatial grid per level.
    For input (B, C, H, W), levels=(1,2,4), returns a list of tokens:
    - level 1: (B, 1, 1, C)
    - level 2: (B, 2, 2, C)
    - level 4: (B, 4, 4, C)
    """
    def __init__(self, levels=(1, 2, 4), pool_type='avg'):
        super().__init__()
        self.levels = levels
        assert pool_type in ['avg', 'max']
        self.pool_type = pool_type

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens_per_level = []
        for l in self.levels:
            if self.pool_type == 'avg':
                pooled = F.adaptive_avg_pool2d(x, (l, l))  # (B, C, l, l)
            else:
                pooled = F.adaptive_max_pool2d(x, (l, l))
            # to (B, l, l, C)
            tokens = pooled.permute(0, 2, 3, 1).contiguous()
            tokens_per_level.append(tokens)
        return tokens_per_level


class WindowAttention(nn.Module):
    """
    Lightweight local window self-attention using MultiheadAttention.
    Operates on tokens shaped (B, Lh, Lw, C) by partitioning into windows of size (wh, ww).
    """
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, window_size):
        # x: (B, Lh, Lw, C), window_size: (wh, ww)
        B, Lh, Lw, C = x.shape
        wh, ww = window_size
        # If window doesn't divide grid, fall back to full-grid attention
        if (Lh % wh != 0) or (Lw % ww != 0):
            seq = x.view(B, Lh * Lw, C)
            out, _ = self.attn(seq, seq, seq)
            out = self.proj_drop(out)
            return out.view(B, Lh, Lw, C)
        # Partition windows
        x_view = x.view(B, Lh // wh, wh, Lw // ww, ww, C)  # (B, Nh, wh, Nw, ww, C)
        windows = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B * (Lh // wh) * (Lw // ww), wh * ww, C)
        # Attention per window
        out, _ = self.attn(windows, windows, windows)
        out = self.proj_drop(out)
        # Merge windows back
        out = out.view(B, Lh // wh, Lw // ww, wh, ww, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, Lh, Lw, C)
        return out


class TransformerMLP(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowTransformerBlock(nn.Module):
    """
    One block: LN -> WindowAttention -> Residual, then LN -> MLP -> Residual.
    Works on tokens (B, Lh, Lw, C).
    """
    def __init__(self, dim=512, num_heads=8, mlp_ratio=2.0, dropout=0.1, window_size=(2, 2)):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = TransformerMLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        # x: (B, Lh, Lw, C)
        B, Lh, Lw, C = x.shape
        # Pre-norm
        x_norm = self.norm1(x.view(B, Lh * Lw, C)).view(B, Lh, Lw, C)
        x_attn = self.attn(x_norm, self.window_size)
        x = x + x_attn
        x_norm2 = self.norm2(x.view(B, Lh * Lw, C)).view(B, Lh, Lw, C)
        x_mlp = self.mlp(x_norm2.view(B, Lh * Lw, C)).view(B, Lh, Lw, C)
        x = x + x_mlp
        return x


class BranchLocalWindowEnhancer(nn.Module):
    """
    Enhance per-branch SPP tokens with local window transformer per level.
    Levels handled separately with level-specific window sizes.
    """
    def __init__(self, dim=512, dropout=0.1):
        super().__init__()
        # Blocks per level: choose window sizes suitable for each grid
        self.block_l1 = None  # (1,1) no need attention
        self.block_l2 = WindowTransformerBlock(dim=dim, num_heads=8, mlp_ratio=2.0, dropout=dropout, window_size=(2, 2))
        self.block_l4 = WindowTransformerBlock(dim=dim, num_heads=8, mlp_ratio=2.0, dropout=dropout, window_size=(2, 2))

    def forward(self, tokens_l1, tokens_l2, tokens_l4):
        # tokens_*: (B, l, l, C)
        # Level 1: optional MLP only or skip
        B, _, _, C = tokens_l1.shape
        # For stability, apply a small MLP without attention on l1
        tokens_l1_flat = tokens_l1.view(B, 1, C)
        tokens_l1_norm = F.layer_norm(tokens_l1_flat, (C,))
        tokens_l1_out = tokens_l1_norm.view(B, 1, 1, C)

        # Level 2 and 4: window attention blocks
        tokens_l2_out = self.block_l2(tokens_l2)
        tokens_l4_out = self.block_l4(tokens_l4)
        return tokens_l1_out, tokens_l2_out, tokens_l4_out


class ResNetBranchWithSPPAndWindow(nn.Module):
    """
    ResNet-18 backbone -> SPP tokens per level -> local window transformer ->
    concat all tokens -> linear projection back to 512-d.
    Interface remains: forward(x) -> (B, 512)
    """
    def __init__(self, backbone: nn.Module, spp_levels=(1, 2, 4), pool_type='avg', dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.spp_tokens = SpatialPyramidPoolingTokens(levels=spp_levels, pool_type=pool_type)
        self.enhancer = BranchLocalWindowEnhancer(dim=512, dropout=dropout)
        total_bins = sum(l * l for l in spp_levels)  # 1 + 4 + 16 = 21
        self.proj = nn.Linear(512 * total_bins, 512)

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
        # SPP tokens per level
        tokens_l1, tokens_l2, tokens_l4 = self.spp_tokens(x)
        # Local window transformer per level
        tokens_l1, tokens_l2, tokens_l4 = self.enhancer(tokens_l1, tokens_l2, tokens_l4)
        # Concatenate all tokens into sequence (B, 21, 512)
        B = x.size(0)
        seq = torch.cat([
            tokens_l1.view(B, 1, 512),
            tokens_l2.view(B, 4, 512),
            tokens_l4.view(B, 16, 512)
        ], dim=1)
        # Project back to 512-d to keep branch interface unchanged
        seq_flat = seq.view(B, -1)  # (B, 21*512)
        out = self.proj(seq_flat)  # (B, 512)
        return out


class LateFusionResNet_SchemeB(nn.Module):
    """
    Late Fusion ResNet with per-branch local window transformer (Scheme B).
    Fusion logic unchanged: concatenate 3 branch features (3*512) -> Dropout -> FC.
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True):
        super().__init__()
        self.rgb_branch = self._create_resnet_branch(in_channels=3, pretrained=pretrained, dropout=0.1)
        self.dynamic_branch = self._create_resnet_branch(in_channels=3, pretrained=pretrained, dropout=0.1)
        self.mhi_branch = self._create_resnet_branch(in_channels=1, pretrained=pretrained, dropout=0.1)
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(1536, num_classes)

    def _create_resnet_branch(self, in_channels, pretrained=True, dropout=0.1):
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
        return ResNetBranchWithSPPAndWindow(model, spp_levels=(1, 2, 4), pool_type='avg', dropout=dropout)

    def forward(self, rgb_input, dynamic_input, mhi_input):
        rgb_features = self.rgb_branch(rgb_input)       # (B, 512)
        dynamic_features = self.dynamic_branch(dynamic_input)  # (B, 512)
        mhi_features = self.mhi_branch(mhi_input)       # (B, 512)
        fused = torch.cat([rgb_features, dynamic_features, mhi_features], dim=1)  # (B, 1536)
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits


def create_fusion_model_schemeB(num_classes=3, dropout_p=0.5, pretrained=True):
    """
    Factory for Scheme B model (per-branch local window transformer).
    """
    return LateFusionResNet_SchemeB(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)