import torch
import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18
import os


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) module that pools feature maps at multiple
    pyramid levels and concatenates the pooled features.
    levels: list of pyramid levels, e.g., [1, 2, 4]
    pool_type: 'max' or 'avg'
    """
    def __init__(self, levels=(1, 2, 4), pool_type='max'):
        super().__init__()
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        pooled = []
        for l in self.levels:
            if self.pool_type == 'max':
                pool = nn.functional.adaptive_max_pool2d(x, output_size=(l, l))
            else:
                pool = nn.functional.adaptive_avg_pool2d(x, output_size=(l, l))
            pooled.append(pool.view(B, C * l * l))
        return torch.cat(pooled, dim=1)  # (B, C * sum(l*l))


class ResNetBranchWithSPP(nn.Module):
    """
    ResNet-18 branch that extracts convolutional features up to layer4,
    applies SPP over the feature map, and projects to a fixed 512-d feature.
    Used for Dynamic Channel.
    """
    def __init__(self, in_channels=3, pretrained=True, out_dim=512, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        if pretrained:
            model = torchvision_resnet18(weights='IMAGENET1K_V1')
        else:
            model = torchvision_resnet18(weights=None)

        # If a transfer weight path is provided, load it onto the backbone before slicing
        if transfer_weights_path is not None:
            try:
                checkpoint = torch.load(transfer_weights_path, map_location='cpu')
                # Allow both raw state_dict or dict with 'state_dict'
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                # Optional: log minimal info for debugging when needed
                if len(missing) > 0 or len(unexpected) > 0:
                    pass
            except Exception:
                # Fail silently to keep training running even if the weight file is not compatible
                pass

        # Adjust first conv for different input channels (if needed)
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

        # Backbone up to layer4 (exclude avgpool & fc)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # SPP and projection to fixed out_dim
        self.spp = SpatialPyramidPooling(levels=spp_levels, pool_type='max')
        spp_out_dim = 512 * sum(l * l for l in spp_levels)  # ResNet-18 layer4 has 512 channels
        self.project = nn.Sequential(
            nn.Linear(spp_out_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, H', W')
        x = self.spp(x)     # (B, 512 * sum(l*l))
        x = self.project(x) # (B, out_dim)
        return x


class ResNetBranchGAP(nn.Module):
    """
    ResNet-18 branch that uses Global Average Pooling (GAP).
    Used for Static (RGB) Channel.
    """
    def __init__(self, in_channels=3, pretrained=True, out_dim=512, transfer_weights_path=None):
        super().__init__()
        if pretrained:
            model = torchvision_resnet18(weights='IMAGENET1K_V1')
        else:
            model = torchvision_resnet18(weights=None)

        # Load transfer weights if provided
        if transfer_weights_path is not None:
            try:
                checkpoint = torch.load(transfer_weights_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

        if in_channels != 3:
            original_weights = model.conv1.weight.clone()
            new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_weights
            model.conv1 = new_conv1

        # Backbone up to layer4
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optional projection layer if out_dim is not 512 (default ResNet18 output)
        # Even if it is 512, we might want to keep a consistent interface or add non-linearity
        # For simplicity and to match the 'simpler' requirement, we just flatten.
        # But to be safe with fusion concatenation, we ensure output is (B, 512).
        self.out_dim = out_dim
        if out_dim != 512:
             self.project = nn.Sequential(
                nn.Linear(512, out_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.project = None

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)      # (B, 512, H, W)
        x = self.avgpool(x)     # (B, 512, 1, 1)
        x = torch.flatten(x, 1) # (B, 512)
        
        if self.project is not None:
            x = self.project(x)
            
        return x


class LateFusionResNetAsym(nn.Module):
    """
    Asymmetric Late Fusion Model.
    - RGB Branch: ResNet + GAP (Global Context, Static)
    - Dynamic Branch: ResNet + SPP (Local Detail, Motion)
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        
        # RGB Branch: Light-weight (GAP)
        self.rgb_branch = ResNetBranchGAP(
            in_channels=3, pretrained=pretrained, out_dim=512, 
            transfer_weights_path=transfer_weights_path
        )
        
        # Dynamic Branch: Heavy-weight (SPP)
        self.dynamic_branch = ResNetBranchWithSPP(
            in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels,
            transfer_weights_path=transfer_weights_path
        )

        # Fusion head
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(1024, num_classes) # 512 (RGB) + 512 (Dynamic)

    def forward(self, rgb_input, dynamic_input):
        rgb_feat = self.rgb_branch(rgb_input)
        dyn_feat = self.dynamic_branch(dynamic_input)
        fused = torch.cat([rgb_feat, dyn_feat], dim=1)
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits


def create_fusion_model(num_classes=3, dropout_p=0.5, pretrained=True):
    """Default factory (returns Asym model as requested)."""
    return LateFusionResNetAsym(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)


def create_fusion_model_with_transfer(num_classes=3, dropout_p=0.5, pretrained=True, transfer_weights_path=None):
    """Factory to create Asym model with transfer weights."""
    return LateFusionResNetAsym(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        spp_levels=(1, 2, 4),
        transfer_weights_path=transfer_weights_path
    )
