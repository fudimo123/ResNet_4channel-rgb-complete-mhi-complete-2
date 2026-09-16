import torch
import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18


class LateFusionResNet(nn.Module):
    """
    Late Fusion ResNet model that combines RGB and Dynamic Image features
    at the feature level before the final classification layer.
    """
    
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True):
        super(LateFusionResNet, self).__init__()
        
        # RGB branch (3 channels)
        self.rgb_branch = self._create_resnet_branch(in_channels=3, pretrained=pretrained)
        
        # Dynamic Image branch (3 channels)
        self.dynamic_branch = self._create_resnet_branch(in_channels=3, pretrained=pretrained)
        
        # Fusion layer
        # Each branch outputs 512 features, so combined is 1024
        self.fusion_dropout = nn.Dropout(p=dropout_p)
        self.fusion_fc = nn.Linear(1024, num_classes)
        
    def _create_resnet_branch(self, in_channels, pretrained=True):
        """
        Create a ResNet-18 branch without the final classification layer
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
            
            # Remove the final classification layer
            model.fc = nn.Identity()
        else:
            raise NotImplementedError("Non-pretrained ResNet is not implemented for fusion model")
            
        return model
    
    def forward(self, rgb_input, dynamic_input):
        """
        Forward pass through both branches and fusion
        
        Args:
            rgb_input: RGB images tensor (batch_size, 3, H, W)
            dynamic_input: Dynamic images tensor (batch_size, 3, H, W)
        
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Extract features from both branches
        rgb_features = self.rgb_branch(rgb_input)  # (batch_size, 512)
        dynamic_features = self.dynamic_branch(dynamic_input)  # (batch_size, 512)
        
        # Concatenate features for late fusion
        fused_features = torch.cat([rgb_features, dynamic_features], dim=1)  # (batch_size, 1024)
        
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