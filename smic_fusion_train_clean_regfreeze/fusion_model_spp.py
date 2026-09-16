import torch
import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18, ResNet18_Weights


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) over convolutional feature maps.
    Uses adaptive pooling to produce fixed-length outputs for levels (e.g., 1,2,4).
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
    ResNet-18 branch that extracts features up to layer4, applies SPP,
    and projects to a fixed out_dim (default 512).
    Supports optional transfer weight loading.
    """
    def __init__(self, in_channels=3, pretrained=True, out_dim=512, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        if pretrained:
            model = torchvision_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = torchvision_resnet18(weights=None)

        # Load transfer weights into the backbone if provided (strict=False to ignore head mismatches)
        if transfer_weights_path is not None:
            try:
                checkpoint = torch.load(transfer_weights_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                # Fail silently if incompatible; backbone will remain ImageNet-pretrained
                pass

        # Adjust first conv for different input channels
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

        # Backbone up to layer4
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # SPP and projection
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


class LateFusionResNetWithSPP(nn.Module):
    """
    Late Fusion model with two ResNet-18+SPP branches (RGB & Dynamic),
    each producing 512-d features, concatenated to 1024-d and classified.
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        self.rgb_branch = ResNetBranchWithSPP(
            in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels,
            transfer_weights_path=transfer_weights_path
        )
        self.dynamic_branch = ResNetBranchWithSPP(
            in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels,
            transfer_weights_path=transfer_weights_path
        )
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(1024, num_classes)

    def forward(self, rgb_input, dynamic_input):
        rgb_feat = self.rgb_branch(rgb_input)
        dyn_feat = self.dynamic_branch(dynamic_input)
        fused = torch.cat([rgb_feat, dyn_feat], dim=1)
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits


def create_fusion_model(num_classes=3, dropout_p=0.5, pretrained=True):
    return LateFusionResNetWithSPP(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)


def create_fusion_model_with_transfer(num_classes=3, dropout_p=0.5, pretrained=True, transfer_weights_path=None):
    return LateFusionResNetWithSPP(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        spp_levels=(1, 2, 4),
        transfer_weights_path=transfer_weights_path
    )