import torch
import torch.nn as nn
from torchvision.models import resnet18
import time
import os
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("THOP not installed, FLOPs calculation might be skipped or less accurate.")

# ==============================================================================
# Model Definition (Copied from export_netron_model.py for self-containment)
# ==============================================================================

class SpatialPyramidPooling(nn.Module):
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
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = avg_out + max_out
        channel_attn = self.sigmoid_channel(channel_out)
        x = x * channel_attn
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv_spatial(spatial_out)
        spatial_attn = self.sigmoid_spatial(spatial_out)
        x = x * spatial_attn
        return x

class SEBlock(nn.Module):
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

class ResNetBranchWithSPP(nn.Module):
    def __init__(self, in_channels=3, out_dim=512, spp_levels=(1, 2, 4)):
        super().__init__()
        model = resnet18(weights=None) 
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        self.cbam = CBAMBlock(in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
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
        x = self.cbam(x)        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.project(x)
        return x

class LateFusionResNetSE(nn.Module):
    def __init__(self, num_classes=3, dropout_p=0.5):
        super().__init__()
        self.rgb_branch = ResNetBranchGrid_CBAM(in_channels=3, out_dim=512)
        self.dynamic_branch = ResNetBranchWithSPP(in_channels=3, out_dim=512, spp_levels=(1, 2, 4))
        self.fusion_dim = 1024
        self.se_fusion = SEBlock(channel=self.fusion_dim, reduction=16)
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, rgb_input, dynamic_input):
        rgb_feat = self.rgb_branch(rgb_input)
        dyn_feat = self.dynamic_branch(dynamic_input)
        fused = torch.cat([rgb_feat, dyn_feat], dim=1)
        fused = self.se_fusion(fused)
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits

# ==============================================================================
# Metric Calculation
# ==============================================================================

def calculate_metrics():
    # 1. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LateFusionResNetSE(num_classes=3)
    model.to(device)
    model.eval()

    # 2. Dummy Inputs (Standard ImageNet size)
    input_size = (1, 3, 224, 224)
    rgb_input = torch.randn(input_size).to(device)
    dyn_input = torch.randn(input_size).to(device)

    # 3. Calculate Params
    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6

    # 4. Calculate FLOPs (using thop)
    flops_g = 0.0
    if THOP_AVAILABLE:
        try:
            # thop.profile returns (flops, params)
            # It handles tuple inputs for models with multiple inputs
            flops, _ = profile(model, inputs=(rgb_input, dyn_input), verbose=False)
            flops_g = flops / 1e9
        except Exception as e:
            print(f"Error calculating FLOPs with thop: {e}")
    else:
        print("THOP not available, skipping FLOPs.")

    # 5. Calculate FPS (Inference Speed)
    # Warm-up
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(rgb_input, dyn_input)
    
    # Measure
    iterations = 100
    print(f"Measuring FPS over {iterations} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(rgb_input, dyn_input)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_img = total_time / iterations
    fps = 1 / avg_time_per_img

    # 6. Format Output
    output_str = (
        "==================================================\n"
        "Efficiency Metrics for Asymmetric Dual-Stream Network\n"
        "==================================================\n"
        f"Model: LateFusionResNetSE (ResNet18 Backbone)\n"
        f"Input Size: {input_size[2]}x{input_size[3]}\n"
        f"Device: {device}\n"
        "--------------------------------------------------\n"
        f"Parameters (Params): {params_m:.2f} M\n"
        f"FLOPs: {flops_g:.2f} G\n"
        f"Inference Speed (FPS): {fps:.2f} frames/sec\n"
        f"Average Latency: {avg_time_per_img*1000:.2f} ms\n"
        "==================================================\n"
    )

    print(output_str)

    # 7. Save to file
    output_dir = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, "efficiency_metrics.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output_str)
    
    print(f"Metrics saved to: {file_path}")

if __name__ == "__main__":
    calculate_metrics()
