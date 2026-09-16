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
    filename_candidates = [f"{model_name}.safetensors"]
    if model_name == "deit_tiny_patch16_224":
        filename_candidates.append("model.safetensors")

    search_dirs = [os.getcwd(), os.path.dirname(__file__)]
    for search_dir in search_dirs:
        for filename in filename_candidates:
            candidate = os.path.join(search_dir, filename)
            if os.path.exists(candidate):
                return candidate
    return None


def _try_build_timm_backbone_from_local_weights(model_name):
    local_weight_path = _find_local_timm_weights(model_name)
    if local_weight_path is None:
        return None

    if load_safetensors_file is None:
        warnings.warn(
            f"Found local weights for '{model_name}' at '{local_weight_path}', "
            "but safetensors is not installed. Skipping local weight loading."
        )
        return None

    model = timm.create_model(model_name, pretrained=False)
    state_dict = load_safetensors_file(local_weight_path)
    compatible_state_dict, skipped_keys = _filter_compatible_state_dict(model, state_dict)
    model.load_state_dict(compatible_state_dict, strict=False)
    print(
        f"[Info] Loaded local safetensors weights for '{model_name}' from '{local_weight_path}'. "
        f"Skipped {len(skipped_keys)} incompatible keys."
    )
    return model


class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels=(1, 2, 4), pool_type="max"):
        super().__init__()
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        pooled_outputs = []
        for level in self.levels:
            if self.pool_type == "max":
                pooled = nn.functional.adaptive_max_pool2d(x, output_size=(level, level))
            else:
                pooled = nn.functional.adaptive_avg_pool2d(x, output_size=(level, level))
            pooled_outputs.append(pooled.view(batch_size, channels * level * level))
        return torch.cat(pooled_outputs, dim=1)


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


class SingleResNet18SPP(nn.Module):
    """
    Single-channel version of the ADF-Net dynamic branch:
    ResNet18 backbone -> remove GAP -> SPP -> classifier.
    """

    def __init__(
        self,
        num_classes=3,
        dropout_p=0.5,
        pretrained=True,
        transfer_weights_path=None,
        spp_levels=(1, 2, 4),
    ):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        _load_transfer_weights(backbone, transfer_weights_path)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.spp = SpatialPyramidPooling(levels=spp_levels, pool_type="max")
        spp_out_dim = 512 * sum(level * level for level in spp_levels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(spp_out_dim, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spp(x)
        return self.classifier(x)


class TimmDropoutClassifier(nn.Module):
    def __init__(self, backbone, num_classes, dropout_p):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.backbone.forward_head(features, pre_logits=True)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = self.dropout(features)
        return self.classifier(features)


def _build_torchvision_model(model_name, num_classes, pretrained, dropout_p, transfer_weights_path=None):
    if model_name == "plain_resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        _load_transfer_weights(model, transfer_weights_path)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        return model

    if model_name == "improved_resnet18_spp":
        return SingleResNet18SPP(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=transfer_weights_path,
        )

    if model_name == "shufflenet_v2_x1_0":
        weights = tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = tv_models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(model.fc.in_features, num_classes))
        return model

    if model_name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        return model

    if model_name == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        return model

    if model_name == "swin_t":
        weights = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
        model = tv_models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        return model

    if model_name == "vit_b_16":
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        return model

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout_p=dropout_p)

    raise ValueError(f"Unsupported torchvision model: {model_name}")


def _build_timm_model(model_name, num_classes, pretrained, dropout_p):
    if timm is None:
        raise ImportError(
            f"Model '{model_name}' requires timm, but timm is not installed in the current environment."
        )

    backbone = None
    if pretrained:
        backbone = _try_build_timm_backbone_from_local_weights(model_name)
        if backbone is None:
            try:
                backbone = timm.create_model(model_name, pretrained=True)
            except Exception as exc:
                warnings.warn(
                    f"Failed to load pretrained weights for timm model '{model_name}'. "
                    f"Falling back to random initialization. Original error: {exc}"
                )
                print(
                    f"[Warning] timm pretrained weights unavailable for '{model_name}'. "
                    "Falling back to pretrained=False."
                )

    if backbone is None:
        backbone = timm.create_model(model_name, pretrained=False)

    backbone.reset_classifier(0)
    return TimmDropoutClassifier(backbone=backbone, num_classes=num_classes, dropout_p=dropout_p)


def create_single_dynamic_model(
    model_name,
    num_classes=3,
    dropout_p=0.5,
    pretrained=True,
    transfer_weights_path=None,
):
    torchvision_models = {
        "simple_cnn",
        "plain_resnet18",
        "improved_resnet18_spp",
        "shufflenet_v2_x1_0",
        "mobilenet_v3_small",
        "efficientnet_b0",
        "swin_t",
        "vit_b_16",
    }
    timm_models = {"deit_tiny_patch16_224"}

    if model_name in torchvision_models:
        return _build_torchvision_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_p=dropout_p,
            transfer_weights_path=transfer_weights_path,
        )

    if model_name in timm_models:
        return _build_timm_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_p=dropout_p,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def available_backbones():
    names = [
        "simple_cnn",
        "plain_resnet18",
        "improved_resnet18_spp",
        "shufflenet_v2_x1_0",
        "mobilenet_v3_small",
        "efficientnet_b0",
        "deit_tiny_patch16_224",
        "swin_t",
        "vit_b_16",
    ]
    if timm is None:
        return [name for name in names if name != "deit_tiny_patch16_224"]
    return names
