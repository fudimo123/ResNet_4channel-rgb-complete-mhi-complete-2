import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as torchvision_resnet18
from fusion_model_grid_cbam import ResNetBranchGrid_CBAM, ResNetBranchWithSPP

class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two feature vectors.
    
    References:
    - Yang Gao, et al. "Compact Bilinear Pooling." CVPR 2016.
    - PyTorch implementation adapted for pure python execution without CUDA extensions.
    """
    def __init__(self, input_dim1, input_dim2, output_dim=8192):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        # Generate Count Sketch projection matrices (fixed, not learnable)
        # h1, h2: index mapping
        # s1, s2: sign mapping (+1 or -1)
        
        # For feature 1
        self.register_buffer('h1', torch.randint(0, output_dim, (input_dim1,)))
        self.register_buffer('s1', torch.randint(0, 2, (input_dim1,)).float() * 2 - 1)
        
        # For feature 2
        self.register_buffer('h2', torch.randint(0, output_dim, (input_dim2,)))
        self.register_buffer('s2', torch.randint(0, 2, (input_dim2,)).float() * 2 - 1)

    def forward(self, x, y):
        # x: (B, input_dim1)
        # y: (B, input_dim2)
        batch_size = x.size(0)
        
        # Sketching
        # We project x and y into the output_dim space using Count Sketch
        
        # Efficient implementation using scatter_add
        # Initialize output vectors
        px = torch.zeros(batch_size, self.output_dim, device=x.device)
        py = torch.zeros(batch_size, self.output_dim, device=y.device)
        
        # Expand indices for batch
        h1_expanded = self.h1.unsqueeze(0).expand(batch_size, -1)
        h2_expanded = self.h2.unsqueeze(0).expand(batch_size, -1)
        
        # Apply signs
        x_signed = x * self.s1
        y_signed = y * self.s2
        
        # Scatter add (Project to output_dim)
        px.scatter_add_(1, h1_expanded, x_signed)
        py.scatter_add_(1, h2_expanded, y_signed)
        
        # Convolution theorem: Multiplication in frequency domain = Convolution in time domain
        # Count Sketch approximates polynomial kernel -> We can use FFT
        # But for bilinear pooling, we just need element-wise product in frequency domain if we consider circular convolution
        # However, the standard CBP uses FFT to compute convolution of the two sketched vectors.
        
        # FFT
        px_fft = torch.fft.rfft(px, dim=1)
        py_fft = torch.fft.rfft(py, dim=1)
        
        # Element-wise product in frequency domain
        out_fft = px_fft * py_fft
        
        # Inverse FFT
        out = torch.fft.irfft(out_fft, n=self.output_dim, dim=1)
        
        # Signed Square Root Normalization (Power Normalization)
        out = torch.sign(out) * torch.sqrt(torch.abs(out) + 1e-12)
        
        # L2 Normalization
        out = F.normalize(out, p=2, dim=1)
        
        return out

class LateFusionResNetCBP(nn.Module):
    """
    Asymmetric Late Fusion Model with Compact Bilinear Pooling.
    - RGB Branch: ResNet + CBAM + Grid Pooling(2x2)
    - Dynamic Branch: ResNet + SPP
    - Fusion: CBP (8192 dim)
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        
        # RGB Branch: Grid(2x2) + CBAM
        # Output dim = 2048
        self.rgb_branch = ResNetBranchGrid_CBAM(
            in_channels=3, pretrained=pretrained, out_dim=512, 
            transfer_weights_path=transfer_weights_path
        )
        # Note: ResNetBranchGrid_CBAM internally projects to out_dim (512).
        # We want the raw grid features before projection? 
        # Actually, 512 is fine. 2048 -> 512 projection is a good bottleneck.
        # Let's keep using the 512 output from the branches.
        # If you want raw 2048, we would need to modify the branch class.
        # But 512 * 512 interaction is already very rich.
        
        # Dynamic Branch: SPP
        # Output dim = 512
        self.dynamic_branch = ResNetBranchWithSPP(
            in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels,
            transfer_weights_path=transfer_weights_path
        )

        # CBP Fusion
        # Inputs are both 512 dim
        self.cbp = CompactBilinearPooling(input_dim1=512, input_dim2=512, output_dim=8192)

        # Fusion head
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(8192, num_classes)

    def forward(self, rgb_input, dynamic_input):
        rgb_feat = self.rgb_branch(rgb_input)       # (B, 512)
        dyn_feat = self.dynamic_branch(dynamic_input) # (B, 512)
        
        # Apply Compact Bilinear Pooling
        fused = self.cbp(rgb_feat, dyn_feat)        # (B, 8192)
        
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits


def create_fusion_model_cbp(num_classes=3, dropout_p=0.5, pretrained=True, transfer_weights_path=None):
    """Factory to create CBP Fusion model."""
    return LateFusionResNetCBP(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        spp_levels=(1, 2, 4),
        transfer_weights_path=transfer_weights_path
    )
