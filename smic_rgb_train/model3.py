import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_p=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MultiScaleChannelAttention(nn.Module):
    """Enhanced channel attention with multi-scale feature aggregation"""
    def __init__(self, in_planes, ratio=8):
        super(MultiScaleChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Multi-scale pooling
        self.pool_3x3 = nn.AdaptiveAvgPool2d(3)
        self.pool_5x5 = nn.AdaptiveAvgPool2d(5)
        
        hidden_dim = max(in_planes // ratio, 8)
        
        self.fc1 = nn.Conv2d(in_planes, hidden_dim, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_dim, in_planes, 1, bias=False)
        
        # Multi-scale feature fusion
        self.conv_fusion = nn.Conv2d(in_planes * 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Global pooling features
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        # Multi-scale pooling features
        pool_3x3 = F.adaptive_avg_pool2d(self.pool_3x3(x), (1, 1))
        pool_5x5 = F.adaptive_avg_pool2d(self.pool_5x5(x), (1, 1))
        
        # Concatenate all features
        combined = torch.cat([avg_out, max_out, pool_3x3, pool_5x5], dim=1)
        
        # Feature fusion
        fused = self.conv_fusion(combined)
        
        # Channel attention
        out = self.fc1(fused)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return self.sigmoid(out)


class EnhancedSpatialAttention(nn.Module):
    """Enhanced spatial attention with dilated convolutions"""
    def __init__(self, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Multi-scale spatial attention
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv_dilated = nn.Conv2d(2, 1, kernel_size, padding=padding*2, dilation=2, bias=False)
        self.conv_fusion = nn.Conv2d(2, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        # Multi-scale spatial features
        out1 = self.conv1(spatial_input)
        out2 = self.conv_dilated(spatial_input)
        
        # Fusion
        combined = torch.cat([out1, out2], dim=1)
        out = self.conv_fusion(combined)
        
        return self.sigmoid(out)


class EnhancedCBAM(nn.Module):
    """Enhanced CBAM with improved attention mechanisms"""
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(EnhancedCBAM, self).__init__()
        self.channel_attention = MultiScaleChannelAttention(in_planes, ratio)
        self.spatial_attention = EnhancedSpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block with improved design"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(channel // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualAttentionBlock(nn.Module):
    """Residual block with integrated attention mechanisms"""
    expansion = 1  # Add expansion attribute
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_p=0.1):
        super(ResidualAttentionBlock, self).__init__()
        self.basic_block = BasicBlock(inplanes, planes, stride, downsample, dropout_p)
        self.cbam = EnhancedCBAM(planes, ratio=8)
        self.se = SEBlock(planes, reduction=8)
        
    def forward(self, x):
        out = self.basic_block(x)
        out = self.cbam(out)
        out = self.se(out)
        return out


class EnhancedResNet(nn.Module):
    """Enhanced ResNet with improved attention and regularization"""

    def __init__(self, block, layers, num_classes=1000, in_channels=3, dropout_p=0.4):
        self.inplanes = 64
        super(EnhancedResNet, self).__init__()
        
        # Enhanced stem with better feature extraction
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with attention
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_p=dropout_p*0.5)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_p=dropout_p*0.6)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_p=dropout_p*0.7)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_p=dropout_p*0.8)
        
        # Global feature aggregation
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Enhanced classifier with multiple pathways
        feature_dim = 512 * block.expansion
        self.feature_fusion = nn.Conv1d(2, 1, 1)  # Fuse avg and max pooled features
        
        # Multi-layer classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.7),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.8),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.9),
            nn.Linear(128, num_classes)
        )
        
        # Auxiliary classifier for regularization
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256 * block.expansion, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dropout_p=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if block == ResidualAttentionBlock:
            layers.append(block(self.inplanes, planes, stride, downsample, dropout_p))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if block == ResidualAttentionBlock:
                layers.append(block(self.inplanes, planes, dropout_p=dropout_p))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_aux=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Get auxiliary output from layer3 for regularization
        aux_out = None
        if return_aux and self.training:
            aux_out = self.aux_classifier(x)
        
        x = self.layer4(x)

        # Enhanced global feature aggregation
        avg_pool = self.global_avgpool(x).flatten(1)  # Shape: (batch_size, 512)
        max_pool = self.global_maxpool(x).flatten(1)  # Shape: (batch_size, 512)
        
        # Fuse features
        pooled_features = torch.stack([avg_pool, max_pool], dim=1)  # Shape: (batch_size, 2, 512)
        fused_features = self.feature_fusion(pooled_features).squeeze(1)  # Shape: (batch_size, 512)
        
        # Classification
        out = self.classifier(fused_features)
        
        if return_aux and aux_out is not None:
            return out, aux_out
        return out


# Model creation functions
def resnet18_enhanced(pretrained=False, **kwargs):
    """Enhanced ResNet-18 with attention mechanisms"""
    model = EnhancedResNet(ResidualAttentionBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_enhanced(pretrained=False, **kwargs):
    """Enhanced ResNet-34 with attention mechanisms"""
    model = EnhancedResNet(ResidualAttentionBlock, [3, 4, 6, 3], **kwargs)
    return model


# Backward compatibility functions
def resnet18(pretrained=False, **kwargs):
    """Constructs an enhanced ResNet-18 model"""
    return resnet18_enhanced(pretrained=pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    """Constructs an enhanced ResNet-34 model"""
    return resnet34_enhanced(pretrained=pretrained, **kwargs)