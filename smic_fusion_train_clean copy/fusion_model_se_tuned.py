import torch
import torch.nn as nn
from fusion_model_grid_cbam import ResNetBranchGrid_CBAM, ResNetBranchWithSPP

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Feature Fusion.
    Adaptive re-weighting of the concatenated feature vector.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze is implicitly done since input is already a feature vector (B, C)
        # We only implement Excitation part
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

class LateFusionResNetSE_Tuned(nn.Module):
    """
    Asymmetric Late Fusion Model with SE-Fusion (Tuned).
    - RGB Branch: ResNet + CBAM + Grid Pooling(2x2) -> 512
    - Dynamic Branch: ResNet + SPP -> 512
    - Fusion: Concat -> SE Block (reduction=4) -> FC
    """
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True, spp_levels=(1, 2, 4), transfer_weights_path=None):
        super().__init__()
        
        # RGB Branch: Grid(2x2) + CBAM
        # Output dim = 512 (projected from 2048)
        self.rgb_branch = ResNetBranchGrid_CBAM(
            in_channels=3, pretrained=pretrained, out_dim=512, 
            transfer_weights_path=transfer_weights_path
        )
        
        # Dynamic Branch: SPP
        # Output dim = 512 (projected from 10752)
        self.dynamic_branch = ResNetBranchWithSPP(
            in_channels=3, pretrained=pretrained, out_dim=512, spp_levels=spp_levels,
            transfer_weights_path=transfer_weights_path
        )

        # SE Fusion Block
        # Input dim = 512 + 512 = 1024
        self.fusion_dim = 1024
        # TUNING: Changed reduction from 16 to 4 to preserve more inter-channel non-linear interactions
        self.se_fusion = SEBlock(channel=self.fusion_dim, reduction=4)

        # Fusion head
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, rgb_input, dynamic_input):
        rgb_feat = self.rgb_branch(rgb_input)       # (B, 512)
        dyn_feat = self.dynamic_branch(dynamic_input) # (B, 512)
        
        # Concat
        fused = torch.cat([rgb_feat, dyn_feat], dim=1) # (B, 1024)
        
        # Apply SE Attention (Adaptive Weighting)
        fused = self.se_fusion(fused)
        
        fused = self.fusion_dropout(fused)
        logits = self.fusion_fc(fused)
        return logits


def create_fusion_model_se_tuned(num_classes=3, dropout_p=0.5, pretrained=True, transfer_weights_path=None):
    """Factory to create Tuned SE Fusion model."""
    return LateFusionResNetSE_Tuned(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        spp_levels=(1, 2, 4),
        transfer_weights_path=transfer_weights_path
    )
