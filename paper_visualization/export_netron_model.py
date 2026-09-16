import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.onnx
import os

# ==============================================================================
# 1. Basic Modules (SPP, CBAM, SE)
# ==============================================================================

class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) module.
    Levels: 1x1, 2x2, 4x4
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
            # Flatten each level
            pooled.append(pool.view(B, C * l * l))
        # Concatenate all levels: 512*(1+4+16) = 10752
        return torch.cat(pooled, dim=1)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Channel Attention + Spatial Attention.
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
    Squeeze-and-Excitation Block for Feature Fusion (No GAP).
    Input is already a 1D vector (B, C), so we skip GAP.
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
        # x: (B, C)
        y = self.fc(x)
        return x * y


# ==============================================================================
# 2. Backbone Branches
# ==============================================================================

class ResNetBranchWithSPP(nn.Module):
    """
    Dynamic Branch: ResNet-18 + SPP + Projection
    """
    def __init__(self, in_channels=3, out_dim=512, spp_levels=(1, 2, 4)):
        super().__init__()
        # Load ResNet (weights don't matter for graph export, but we use them if available)
        model = resnet18(weights=None) 
        
        # Adjust first conv if input is not 3 channels (optional handling)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.spp = SpatialPyramidPooling(levels=spp_levels, pool_type='max')
        spp_out_dim = 512 * sum(l * l for l in spp_levels) # 512 * 21 = 10752
        
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
    RGB Branch: ResNet-18 + CBAM + Grid Pooling + Projection
    """
    def __init__(self, in_channels=3, out_dim=512):
        super().__init__()
        model = resnet18(weights=None)
        
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # CBAM
        self.cbam = CBAMBlock(in_channels=512)

        # Grid Pooling (2x2)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # 512 * 2 * 2 = 2048
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
        x = self.layer4(x)      
        
        # Attention BEFORE Pooling
        x = self.cbam(x)        
        
        x = self.avgpool(x)     # (B, 512, 2, 2)
        x = torch.flatten(x, 1) # (B, 2048)
        x = self.project(x)     # (B, 512)
        return x


# ==============================================================================
# 3. Full Network
# ==============================================================================

class LateFusionResNetSE(nn.Module):
    """
    Full Network: RGB Branch + Dynamic Branch -> Concat -> SE -> FC
    """
    def __init__(self, num_classes=3, dropout_p=0.5):
        super().__init__()
        
        self.rgb_branch = ResNetBranchGrid_CBAM(in_channels=3, out_dim=512)
        self.dynamic_branch = ResNetBranchWithSPP(in_channels=3, out_dim=512, spp_levels=(1, 2, 4))

        self.fusion_dim = 1024
        self.se_fusion = SEBlock(channel=self.fusion_dim, reduction=16)

        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, rgb_input, dynamic_input):
        # RGB Stream
        rgb_feat = self.rgb_branch(rgb_input)       # (B, 512)
        
        # Dynamic Stream
        dyn_feat = self.dynamic_branch(dynamic_input) # (B, 512)
        
        # Fusion
        fused = torch.cat([rgb_feat, dyn_feat], dim=1) # (B, 1024)
        
        # SE Attention
        fused = self.se_fusion(fused)
        
        # Classification
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits


# ==============================================================================
# 4. Export Function
# ==============================================================================

if __name__ == "__main__":
    print("Initializing Model...")
    model = LateFusionResNetSE(num_classes=3, dropout_p=0.5)
    model.eval()

    # --- OPTIONAL: Load Weights ---
    # If you have a specific .pth file you want to check, uncomment and set path below:
    # checkpoint_path = r"D:\path\to\your\checkpoint.pth"
    # if os.path.exists(checkpoint_path):
    #     print(f"Loading weights from {checkpoint_path}")
    #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #     state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    #     model.load_state_dict(state_dict, strict=False)
    # ------------------------------

    print("Creating Dummy Inputs (Using 256x256 to avoid ONNX export issues with AdaptivePool on 7x7 features)...")
    # We use 256x256 instead of 224x224 because 224->7x7 feature map. 
    # AdaptiveAvgPool((2,2)) on 7x7 is not supported by ONNX export (7 is not divisible by 2).
    # 256->8x8 feature map. 8 is divisible by 2.
    rgb_input = torch.randn(1, 3, 256, 256)
    dynamic_input = torch.randn(1, 3, 256, 256)

    output_onnx_path = "fusion_network_structure.onnx"
    
    print(f"Exporting to {output_onnx_path}...")
    torch.onnx.export(
        model, 
        (rgb_input, dynamic_input), 
        output_onnx_path,
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=12,          # ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names = ['RGB_Input', 'Dynamic_Input'],   # the model's input names
        output_names = ['Output_Logits'], # the model's output names
        dynamic_axes={
            'RGB_Input' : {0 : 'batch_size'},    # variable length axes
            'Dynamic_Input' : {0 : 'batch_size'},
            'Output_Logits' : {0 : 'batch_size'}
        }
    )
    print("Success! You can now open 'fusion_network_structure.onnx' in Netron (https://netron.app).")
