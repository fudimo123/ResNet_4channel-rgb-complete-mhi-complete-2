import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as torchvision_resnet18


class SpatialPyramidPooling(nn.Module):
    """
    A simple SPP module using adaptive pooling to aggregate multi-scale features.
    Levels: tuple of output sizes (e.g., (1, 2, 4)).
    Pool type: 'avg' or 'max'.
    Output shape: (B, C * sum(l*l for l in levels)) for input (B, C, H, W).
    """
    def __init__(self, levels=(1, 2, 4), pool_type='avg'):
        super().__init__()
        self.levels = levels
        assert pool_type in ['avg', 'max']
        self.pool_type = pool_type

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        outputs = []
        for l in self.levels:
            if self.pool_type == 'avg':
                pooled = F.adaptive_avg_pool2d(x, (l, l))
            else:
                pooled = F.adaptive_max_pool2d(x, (l, l))
            outputs.append(pooled.view(B, C * l * l))
        return torch.cat(outputs, dim=1)


class ResNetBranchWithSPP(nn.Module):
    """
    Wrap a torchvision ResNet-18 backbone to extract features up to layer4,
    then apply SPP and project back to 512-dim to keep the fusion interface unchanged.
    """
    def __init__(self, backbone: nn.Module, spp_levels=(1, 2, 4), pool_type='avg'):
        super().__init__()
        self.backbone = backbone
        # SPP over layer4 feature map (C=512)
        self.spp = SpatialPyramidPooling(levels=spp_levels, pool_type=pool_type)
        total_bins = sum(l * l for l in spp_levels)
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
        # Multi-scale pooling and projection to 512-dim
        x = self.spp(x)
        x = self.proj(x)  # (B, 512)
        return x


class LateFusionResNet(nn.Module):
    """
    Late Fusion ResNet model that combines RGB, Dynamic Image, and MHI features
    at the feature level before the final classification layer.
    """
    
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True):
        super(LateFusionResNet, self).__init__()
        
        # RGB branch (3 channels)
        self.rgb_branch = self._create_resnet_branch(in_channels=3, pretrained=pretrained)
        
        # Dynamic Image branch (3 channels)
        self.dynamic_branch = self._create_resnet_branch(in_channels=3, pretrained=pretrained)
        
        # MHI branch (1 channel)
        self.mhi_branch = self._create_resnet_branch(in_channels=1, pretrained=pretrained)
        
        # Transformer-based fusion across branches (Scheme A, minimal change)
        # Use branch features (each 512-dim) as tokens in a short sequence of length 3.
        self.branch_pos_embed = nn.Parameter(torch.randn(1, 3, 512) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Fusion layer
        # Each branch outputs 512 features, and we keep the concatenation (3 * 512 = 1536)
        # to preserve the classifier input shape for comparability with previous experiments.
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(1536, num_classes)
        
    def _create_resnet_branch(self, in_channels, pretrained=True):
        """
        Create a ResNet-18 branch without the final classification layer,
        enhanced with an SPP module to achieve multi-scale feature aggregation
        while keeping the branch output dimension at 512.
        """
        if pretrained:
            model = torchvision_resnet18(weights='IMAGENET1K_V1')
            
            # Modify the first convolutional layer for different input channels
            if in_channels != 3:
                original_weights = model.conv1.weight.clone()
                new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    if in_channels == 1:
                        # For single channel (MHI), use the mean of RGB weights
                        new_conv1.weight[:, 0, :, :] = torch.mean(original_weights, dim=1)
                    else:
                        # Copy original weights for the first 3 channels
                        new_conv1.weight[:, :3, :, :] = original_weights
                        # Initialize remaining channels by averaging original weights
                        for i in range(3, in_channels):
                            new_conv1.weight[:, i, :, :] = torch.mean(original_weights, dim=1)
                model.conv1 = new_conv1
            
            # Remove the final classification layer (we will use layer4 features)
            model.fc = nn.Identity()
        else:
            raise NotImplementedError("Non-pretrained ResNet is not implemented for fusion model")
            
        # Wrap the backbone with SPP and a projection back to 512-dim
        return ResNetBranchWithSPP(model, spp_levels=(1, 2, 4), pool_type='avg')
    
    def forward(self, rgb_input, dynamic_input, mhi_input):
        """
        Forward pass through all three branches with transformer-based fusion.
        
        Args:
            rgb_input: RGB images tensor (batch_size, 3, H, W)
            dynamic_input: Dynamic images tensor (batch_size, 3, H, W)
            mhi_input: MHI images tensor (batch_size, 1, H, W)
        
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Extract features from all three branches (each: (B, 512))
        rgb_features = self.rgb_branch(rgb_input)
        dynamic_features = self.dynamic_branch(dynamic_input)
        mhi_features = self.mhi_branch(mhi_input)
        
        # Prepare sequence for transformer: (B, 3, 512)
        seq = torch.stack([rgb_features, dynamic_features, mhi_features], dim=1)
        seq = seq + self.branch_pos_embed  # add branch positional embedding
        seq = self.transformer(seq)        # transformer encoder
        
        # Concatenate transformed tokens to keep classifier input shape unchanged: (B, 1536)
        fused_features = seq.reshape(seq.size(0), -1)
        
        # Apply dropout and final classification
        fused_features = self.fusion_dropout(fused_features)
        output = self.fusion_fc(fused_features)
        
        return output


def create_fusion_model(num_classes=3, dropout_p=0.5, pretrained=True):
    """
    Factory function to create a late fusion ResNet model
    
    Args:
        num_classes: Number of output classes
        dropout_p: Dropout probability
        pretrained: Whether to use pretrained weights
    
    Returns:
        LateFusionResNet model
    """
    return LateFusionResNet(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)