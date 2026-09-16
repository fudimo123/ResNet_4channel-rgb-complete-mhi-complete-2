import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torchvision.models import resnet18 as torchvision_resnet18
import os

# ---------------------------------------------------------
# Core Modules (Unchanged from your original architecture)
# ---------------------------------------------------------

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

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Feature Fusion.
    Original version: reduction=16
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

# ---------------------------------------------------------
# Dynamic Branch (Unchanged: ResNet18 + SPP)
# ---------------------------------------------------------

class ResNetBranchWithSPP(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, out_dim=512, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        if pretrained:
            model = torchvision_resnet18(weights='IMAGENET1K_V1')
        else:
            model = torchvision_resnet18(weights=None)

        if transfer_weights_path is not None:
            if os.path.exists(transfer_weights_path):
                print(f"[Dynamic Branch] Loading transfer weights from: {transfer_weights_path}")
                try:
                    checkpoint = torch.load(transfer_weights_path, map_location='cpu')
                    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"[Dynamic Branch] Error loading weights: {e}")

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

# ---------------------------------------------------------
# NEW RGB Branch: InceptionResnetV1 (VGGFace2) + CBAM + GridPool
# ---------------------------------------------------------

class VGGFaceBranchGrid_CBAM(nn.Module):
    """
    InceptionResnetV1 branch for Static (RGB) Channel.
    Feature: CBAM Attention + Grid Pooling (2x2).
    Pretrained on VGGFace2 for robust facial feature extraction.
    """
    def __init__(self, pretrained=True, out_dim=512):
        super().__init__()
        
        # Load InceptionResnetV1 with VGGFace2 weights
        print("[RGB Branch] Loading InceptionResnetV1 pretrained on VGGFace2...")
        vggface_model = InceptionResnetV1(pretrained='vggface2' if pretrained else None)
        
        # We strip the final classification head (logits) and global pooling (avgpool_1a)
        # because we want to apply our own CBAM and Grid Pooling on the spatial feature map.
        # The last spatial feature map in InceptionResnetV1 is before the avgpool_1a.
        # It has 1792 channels.
        
        self.features = nn.Sequential(
            vggface_model.conv2d_1a,
            vggface_model.conv2d_2a,
            vggface_model.conv2d_2b,
            vggface_model.maxpool_3a,
            vggface_model.conv2d_3b,
            vggface_model.conv2d_4a,
            vggface_model.conv2d_4b,
            vggface_model.repeat_1,
            vggface_model.mixed_6a,
            vggface_model.repeat_2,
            vggface_model.mixed_7a,
            vggface_model.repeat_3,
            vggface_model.block8
        )
        
        # Feature map channel dimension is 1792 at this point
        in_channels = 1792
        
        # --- CBAM Module ---
        self.cbam = CBAMBlock(in_channels=in_channels)

        # --- Grid Pooling (2x2) ---
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Output dimension calculation: 1792 channels * 2 * 2 grid = 7168 features
        grid_out_dim = in_channels * 2 * 2 
        
        # Project down to the required 512 dimensions (to match dynamic branch)
        self.project = nn.Sequential(
            nn.Linear(grid_out_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input size (B, 3, 224, 224) -> Output spatial map (B, 1792, 5, 5)
        x = self.features(x)
        
        # Apply Attention BEFORE Pooling
        x = self.cbam(x)        # (B, 1792, H, W)
        
        x = self.avgpool(x)     # (B, 1792, 2, 2)
        x = torch.flatten(x, 1) # (B, 7168)
        
        x = self.project(x)     # (B, 512)
        return x

# ---------------------------------------------------------
# Heterogeneous Late Fusion Model
# ---------------------------------------------------------

class HeteroFusionResNetSE(nn.Module):
    """
    Heterogeneous Asymmetric Late Fusion Model.
    - RGB Branch: InceptionResnetV1 (VGGFace2) + CBAM + Grid Pooling(2x2) -> 512
    - Dynamic Branch: ResNet18 (ImageNet) + SPP -> 512
    - Fusion: Concat -> SE Block (reduction=16) -> FC
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        
        # RGB Branch (VGGFace2)
        self.rgb_branch = VGGFaceBranchGrid_CBAM(
            pretrained=pretrained, out_dim=512
        )
        
        # Dynamic Branch (ImageNet / Custom Transfer)
        self.dynamic_branch = ResNetBranchWithSPP(
            in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels,
            transfer_weights_path=transfer_weights_path
        )

        # SE Fusion Block (Original Reduction=16)
        self.fusion_dim = 1024
        self.se_fusion = SEBlock(channel=self.fusion_dim, reduction=16)

        # Fusion head
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, rgb_input, dynamic_input):
        rgb_feat = self.rgb_branch(rgb_input)       # (B, 512)
        dyn_feat = self.dynamic_branch(dynamic_input) # (B, 512)
        
        # Concat
        fused = torch.cat([rgb_feat, dyn_feat], dim=1) # (B, 1024)
        
        # Apply SE Attention
        fused = self.se_fusion(fused)
        
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits


def create_hetero_fusion_model_vggface(num_classes=3, dropout_p=0.5, pretrained=True, transfer_weights_path=None):
    """Factory to create Heterogeneous VGGFace2 + ResNet Fusion model."""
    return HeteroFusionResNetSE(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        spp_levels=(1, 2, 4),
        transfer_weights_path=transfer_weights_path
    )
