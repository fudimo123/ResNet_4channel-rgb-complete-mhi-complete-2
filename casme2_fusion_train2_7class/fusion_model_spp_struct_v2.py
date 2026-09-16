import torch
import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18


class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels=(1, 2, 3, 4), pool_type='max'):
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


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        avg_pool = nn.functional.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = nn.functional.adaptive_max_pool2d(x, 1).view(B, C)
        channel_att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(B, C, 1, 1)
        x = x * channel_att
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = torch.cat([avg_map, max_map], dim=1)
        spatial_att = torch.sigmoid(self.spatial(spatial_map))
        return x * spatial_att


class ResNetBranchWithSPP(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, out_dim=512, spp_levels=(1, 2, 3, 4), transfer_weights_path=None):
        super().__init__()
        model = torchvision_resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        if transfer_weights_path is not None:
            try:
                checkpoint = torch.load(transfer_weights_path, map_location='cpu')
                state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass
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
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.cbam = ChannelSpatialAttention(channels=512)
        self.spp = SpatialPyramidPooling(levels=spp_levels, pool_type='max')
        spp_out_dim = 512 * sum(l * l for l in spp_levels)
        self.project = nn.Sequential(
            nn.Linear(spp_out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.spp(x)
        x = self.project(x)
        return x


class LateFusionResNetWithSPPStructV2(nn.Module):
    def __init__(self, num_classes=7, dropout_p=0.5, pretrained=True, spp_levels=(1, 2, 3, 4), transfer_weights_path=None):
        super().__init__()
        self.rgb_branch = ResNetBranchWithSPP(in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels, transfer_weights_path=transfer_weights_path)
        self.dynamic_branch = ResNetBranchWithSPP(in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels, transfer_weights_path=transfer_weights_path)
        self.gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
        self.fusion_fc1 = nn.Linear(1024, 512)
        self.fusion_bn1 = nn.BatchNorm1d(512)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc2 = nn.Linear(512, num_classes)

    def forward(self, rgb_input, dynamic_input, return_feat=False):
        rgb_feat = self.rgb_branch(rgb_input)
        dyn_feat = self.dynamic_branch(dynamic_input)
        concat = torch.cat([rgb_feat, dyn_feat], dim=1)
        gates = self.gate(concat)
        w_rgb = gates[:, 0].unsqueeze(1)
        w_dyn = gates[:, 1].unsqueeze(1)
        fused = torch.cat([w_rgb * rgb_feat, w_dyn * dyn_feat], dim=1)
        x = self.fusion_fc1(fused)
        x = self.fusion_bn1(x)
        x = self.fusion_relu(x)
        x = self.fusion_dropout(x)
        logits = self.fusion_fc2(x)
        if return_feat:
            return logits, x
        return logits


def create_fusion_model(num_classes=7, dropout_p=0.5, pretrained=True):
    return LateFusionResNetWithSPPStructV2(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)

def create_fusion_model_with_transfer(num_classes=7, dropout_p=0.5, pretrained=True, transfer_weights_path=None):
    return LateFusionResNetWithSPPStructV2(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained, spp_levels=(1, 2, 3, 4), transfer_weights_path=transfer_weights_path)

