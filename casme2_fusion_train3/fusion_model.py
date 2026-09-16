import torch
import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18


class LateFusionResNet(nn.Module):
    """
    加权投票的晚期融合 ResNet 模型：RGB、Dynamic、MHI 三个分支各自输出 logits，
    在 logits 级进行加权求和实现加权投票融合，最终用于分类。
    """
    
    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True):
        super(LateFusionResNet, self).__init__()
        
        # RGB branch (3 channels)
        self.rgb_branch = self._create_resnet_branch(in_channels=3, pretrained=pretrained)
        
        # Dynamic Image branch (3 channels)
        self.dynamic_branch = self._create_resnet_branch(in_channels=3, pretrained=pretrained)
        
        # MHI branch (1 channel)
        self.mhi_branch = self._create_resnet_branch(in_channels=1, pretrained=pretrained)
        
        # 每个分支独立的分类头（512 -> num_classes），并在分类前使用dropout
        self.branch_dropout = nn.Dropout(p=dropout_p)
        self.rgb_classifier = nn.Linear(512, num_classes)
        self.dynamic_classifier = nn.Linear(512, num_classes)
        self.mhi_classifier = nn.Linear(512, num_classes)
        
        # 加权投票的权重，默认等权重（1,1,1），并在运行时进行归一化
        self.register_buffer('fusion_weights', torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

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
    
    def forward(self, rgb_input, dynamic_input, mhi_input):
        """
        Forward pass through all three branches with weighted voting at logits level
        
        Args:
            rgb_input: RGB images tensor (batch_size, 3, H, W)
            dynamic_input: Dynamic images tensor (batch_size, 3, H, W)
            mhi_input: MHI images tensor (batch_size, 1, H, W)
        
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # 计算各分支特征
        rgb_features = self.rgb_branch(rgb_input)  # (batch_size, 512)
        dynamic_features = self.dynamic_branch(dynamic_input)  # (batch_size, 512)
        mhi_features = self.mhi_branch(mhi_input)  # (batch_size, 512)
        
        # 分别进行dropout并通过各自分类头得到logits
        rgb_logits = self.rgb_classifier(self.branch_dropout(rgb_features))
        dynamic_logits = self.dynamic_classifier(self.branch_dropout(dynamic_features))
        mhi_logits = self.mhi_classifier(self.branch_dropout(mhi_features))
        
        # 计算加权系数并进行加权投票（logits级加权求和）
        weights = self.fusion_weights / (self.fusion_weights.sum() + 1e-8)
        fused_logits = (
            weights[0] * rgb_logits +
            weights[1] * dynamic_logits +
            weights[2] * mhi_logits
        )
        
        return fused_logits


def create_fusion_model(num_classes=3, dropout_p=0.5, pretrained=True):
    """
    Factory function to create a weighted voting late fusion ResNet model
    
    Args:
        num_classes: Number of output classes
        dropout_p: Dropout probability
        pretrained: Whether to use pretrained weights
    
    Returns:
        LateFusionResNet model (weighted voting at logits level)
    """
    return LateFusionResNet(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)