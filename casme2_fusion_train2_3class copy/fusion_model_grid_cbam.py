import torch
import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18
import os

class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) module.
    """
    def __init__(self, levels=(1, 2, 4), pool_type='max'):
        super().__init__()
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        B, C, H, W = x.size()
        pooled = []
        for l in self.levels:
            if self.pool_type == 'max':
                pool = nn.functional.adaptive_max_pool2d(x, output_size=(l, l))
            else:
                pool = nn.functional.adaptive_avg_pool2d(x, output_size=(l, l))
            pooled.append(pool.view(B, C * l * l))
        return torch.cat(pooled, dim=1)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Consists of Channel Attention + Spatial Attention.
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = avg_out + max_out
        channel_attn = self.sigmoid_channel(channel_out)
        x = x * channel_attn

        # 2. Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv_spatial(spatial_out)
        spatial_attn = self.sigmoid_spatial(spatial_out)
        x = x * spatial_attn

        return x


class ResNetBranchWithSPP(nn.Module):
    """
    ResNet-18 branch for Dynamic Channel (SPP enabled).
    """
    def __init__(self, in_channels=3, pretrained=True, out_dim=512, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        if pretrained:
            model = torchvision_resnet18(weights='IMAGENET1K_V1')
        else:
            model = torchvision_resnet18(weights=None)

        if transfer_weights_path is not None:
            if os.path.exists(transfer_weights_path):
                print(f"[Model Init] Loading transfer weights from: {transfer_weights_path}")
                try:
                    checkpoint = torch.load(transfer_weights_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"[Model Init] Error loading weights: {e}")
            else:
                print(f"[Model Init] Weight file NOT FOUND at: {transfer_weights_path}. Using ImageNet weights instead.")

        if in_channels != 3:
            original_weights = model.conv1.weight.clone()
            new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_weights
            model.conv1 = new_conv1

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.spp = SpatialPyramidPooling(levels=spp_levels, pool_type='max')
        spp_out_dim = 512 * sum(l * l for l in spp_levels)
        self.project = nn.Sequential(
            nn.Linear(spp_out_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spp(x)
        x = self.project(x)
        return x


class ResNetBranchGrid_CBAM(nn.Module):
    """
    ResNet-18 branch for Static (RGB) Channel.
    Feature: CBAM Attention + Grid Pooling (2x2).
    """
    def __init__(self, in_channels=3, pretrained=True, out_dim=512, transfer_weights_path=None):
        super().__init__()
        if pretrained:
            model = torchvision_resnet18(weights='IMAGENET1K_V1')
        else:
            model = torchvision_resnet18(weights=None)

        if transfer_weights_path is not None:
            if os.path.exists(transfer_weights_path):
                print(f"[Model Init] Loading transfer weights from: {transfer_weights_path}")
                try:
                    checkpoint = torch.load(transfer_weights_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"[Model Init] Error loading weights: {e}")
            else:
                print(f"[Model Init] Weight file NOT FOUND at: {transfer_weights_path}. Using ImageNet weights instead.")

        if in_channels != 3:
            original_weights = model.conv1.weight.clone()
            new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_weights
            model.conv1 = new_conv1

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # --- CBAM Module ---
        self.cbam = CBAMBlock(in_channels=512)

        # --- Grid Pooling (2x2) ---
        # Instead of 1x1, we keep 2x2 spatial grid
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Output dimension calculation
        # 512 channels * 2 * 2 grid = 2048 features
        grid_out_dim = 512 * 2 * 2 
        
        self.project = nn.Sequential(
            nn.Linear(grid_out_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)      # (B, 512, H, W)
        
        # Apply Attention BEFORE Pooling
        x = self.cbam(x)        # (B, 512, H, W) - refined features
        
        x = self.avgpool(x)     # (B, 512, 2, 2) - grid pooling
        x = torch.flatten(x, 1) # (B, 2048)
        
        x = self.project(x)     # (B, 512)
        return x


class LateFusionResNetAsymGridCBAM(nn.Module):
    """
    Asymmetric Late Fusion Model with Grid+CBAM.
    - RGB Branch: ResNet + CBAM + Grid Pooling(2x2)
    - Dynamic Branch: ResNet + SPP
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        
        # RGB Branch: Grid(2x2) + CBAM
        self.rgb_branch = ResNetBranchGrid_CBAM(
            in_channels=3, pretrained=pretrained, out_dim=512, 
            transfer_weights_path=transfer_weights_path
        )
        
        # Dynamic Branch: SPP
        self.dynamic_branch = ResNetBranchWithSPP(
            in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels,
            transfer_weights_path=transfer_weights_path
        )

        # Fusion head
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(1024, num_classes)

    def forward(self, rgb_input, dynamic_input):
        rgb_feat = self.rgb_branch(rgb_input)
        dyn_feat = self.dynamic_branch(dynamic_input)
        fused = torch.cat([rgb_feat, dyn_feat], dim=1)
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits


def create_fusion_model_with_grid_cbam(num_classes=3, dropout_p=0.5, pretrained=True, transfer_weights_path=None):
    """Factory to create Asym model with Grid+CBAM."""
    return LateFusionResNetAsymGridCBAM(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        spp_levels=(1, 2, 4),
        transfer_weights_path=transfer_weights_path
    )
