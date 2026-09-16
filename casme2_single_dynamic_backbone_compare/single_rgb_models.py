import os
import warnings

import torch
import torch.nn as nn
from torchvision import models as tv_models

try:
    import timm
except ImportError:
    timm = None

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None


def _load_transfer_weights(model, transfer_weights_path):
    if transfer_weights_path is None or not os.path.exists(transfer_weights_path):
        return

    checkpoint = torch.load(transfer_weights_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)


def _filter_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    return compatible, skipped


def _find_local_timm_weights(model_name):
    filename_candidates = []
    if model_name == "deit_tiny_patch16_224":
        filename_candidates.extend(["model.safetensors", f"{model_name}.safetensors"])
    else:
        filename_candidates.extend([f"{model_name}.safetensors", "model.safetensors"])

    search_dirs = [os.getcwd(), os.path.dirname(__file__)]
    for search_dir in search_dirs:
        for filename in filename_candidates:
            candidate = os.path.join(search_dir, filename)
            if os.path.exists(candidate):
                return candidate
    return None


def _try_build_timm_from_local_weights(model_name, num_classes):
    local_weight_path = _find_local_timm_weights(model_name)
    if local_weight_path is None:
        return None

    if load_safetensors_file is None:
        warnings.warn(
            f"Found local weights for '{model_name}' at '{local_weight_path}', "
            "but safetensors is not installed. Skipping local weight loading."
        )
        return None

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = load_safetensors_file(local_weight_path)
    compatible_state_dict, skipped_keys = _filter_compatible_state_dict(model, state_dict)
    model.load_state_dict(compatible_state_dict, strict=False)
    print(
        f"[Info] Loaded local safetensors weights for '{model_name}' from '{local_weight_path}'. "
        f"Skipped {len(skipped_keys)} incompatible keys."
    )
    return model


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_attn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SingleResNet18GridCBAM(nn.Module):
    """
    Single-channel version of the ADF-Net static branch:
    ResNet18 backbone -> remove GAP -> CBAM -> Grid Pooling(2x2) -> classifier.
    """

    def __init__(self, num_classes=3, dropout_p=0.5, pretrained=True, transfer_weights_path=None):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        _load_transfer_weights(backbone, transfer_weights_path)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.cbam = CBAMBlock(in_channels=512)
        self.grid_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(512 * 2 * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.grid_pool(x)
        return self.classifier(x)


def _build_torchvision_model(model_name, num_classes, pretrained, dropout_p, transfer_weights_path=None):
    if model_name == "plain_resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        _load_transfer_weights(model, transfer_weights_path)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        return model

    if model_name == "improved_resnet18_grid_cbam":
        return SingleResNet18GridCBAM(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=transfer_weights_path,
        )

    if model_name == "shufflenet_v2_x1_0":
        weights = tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = tv_models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "swin_t":
        weights = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
        model = tv_models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model

    if model_name == "vit_b_16":
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout_p=dropout_p)

    raise ValueError(f"Unsupported torchvision model: {model_name}")


def _build_timm_model(model_name, num_classes, pretrained):
    if timm is None:
        raise ImportError(
            f"Model '{model_name}' requires timm, but timm is not installed in the current environment."
        )

    local_model = _try_build_timm_from_local_weights(model_name, num_classes)
    if local_model is not None:
        return local_model

    if not pretrained:
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    try:
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    except Exception as exc:
        warnings.warn(
            f"Failed to load pretrained weights for timm model '{model_name}'. "
            f"Falling back to random initialization. Original error: {exc}"
        )
        print(
            f"[Warning] timm pretrained weights unavailable for '{model_name}'. "
            "Falling back to pretrained=False."
        )
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def create_single_rgb_model(
    model_name,
    num_classes=3,
    dropout_p=0.5,
    pretrained=True,
    transfer_weights_path=None,
):
    torchvision_models = {
        "simple_cnn",
        "plain_resnet18",
        "improved_resnet18_grid_cbam",
        "shufflenet_v2_x1_0",
        "mobilenet_v3_small",
        "efficientnet_b0",
        "swin_t",
        "vit_b_16",
    }
    timm_models = {
        "mobilevit_xxs",
        "repvit_m0_9",
        "deit_tiny_patch16_224",
    }

    if model_name in torchvision_models:
        return _build_torchvision_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_p=dropout_p,
            transfer_weights_path=transfer_weights_path,
        )

    if model_name in timm_models:
        return _build_timm_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model_name: {model_name}")


def available_backbones():
    names = [
        "simple_cnn",
        "plain_resnet18",
        "improved_resnet18_grid_cbam",
        "shufflenet_v2_x1_0",
        "mobilenet_v3_small",
        "efficientnet_b0",
        "swin_t",
        "vit_b_16",
    ]
    if timm is not None:
        for name in ["mobilevit_xxs", "repvit_m0_9", "deit_tiny_patch16_224"]:
            if name in timm.list_models():
                names.append(name)
    return names
