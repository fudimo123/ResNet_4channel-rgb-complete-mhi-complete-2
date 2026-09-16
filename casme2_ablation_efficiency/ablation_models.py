from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tv_models


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AblationSpec:
    key: str
    display_name: str
    result_metrics_path: Path
    branch_type: str
    feature_strategy: str
    fusion_strategy: str
    pretraining: str


def _resnet18_backbone(pretrained: bool) -> tv_models.ResNet:
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    return tv_models.resnet18(weights=weights)


class PlainSingleResNet18(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5, pretrained: bool = False):
        super().__init__()
        backbone = _resnet18_backbone(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(in_features, num_classes))
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels: tuple[int, ...] = (1, 2, 4), pool_type: str = "max"):
        super().__init__()
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        pooled_outputs = []
        for level in self.levels:
            if self.pool_type == "max":
                pooled = nn.functional.adaptive_max_pool2d(x, output_size=(level, level))
            else:
                pooled = nn.functional.adaptive_avg_pool2d(x, output_size=(level, level))
            pooled_outputs.append(pooled.view(batch_size, channels * level * level))
        return torch.cat(pooled_outputs, dim=1)


class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_attn


class ResNetBranchPlain(nn.Module):
    def __init__(self, pretrained: bool = False, out_dim: int = 512):
        super().__init__()
        backbone = _resnet18_backbone(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class ResNetBranchWithSPP(nn.Module):
    def __init__(self, pretrained: bool = False, out_dim: int = 512, spp_levels: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        backbone = _resnet18_backbone(pretrained=pretrained)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.spp = SpatialPyramidPooling(levels=spp_levels, pool_type="max")
        spp_out_dim = 512 * sum(level * level for level in spp_levels)
        self.project = nn.Sequential(nn.Linear(spp_out_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spp(x)
        return self.project(x)


class ResNetBranchGridCBAM(nn.Module):
    def __init__(self, pretrained: bool = False, out_dim: int = 512):
        super().__init__()
        backbone = _resnet18_backbone(pretrained=pretrained)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.cbam = CBAMBlock(in_channels=512)
        self.grid_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.project = nn.Sequential(nn.Linear(512 * 2 * 2, out_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.grid_pool(x)
        x = torch.flatten(x, 1)
        return self.project(x)


class LateFusionResNetPlain(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5, pretrained: bool = False):
        super().__init__()
        self.rgb_branch = ResNetBranchPlain(pretrained=pretrained)
        self.dynamic_branch = ResNetBranchPlain(pretrained=pretrained)
        self.fusion_dropout = nn.Dropout(dropout_p)
        self.fusion_fc = nn.Linear(1024, num_classes)

    def forward(self, rgb_input: torch.Tensor, dynamic_input: torch.Tensor) -> torch.Tensor:
        rgb_features = self.rgb_branch(rgb_input)
        dynamic_features = self.dynamic_branch(dynamic_input)
        fused = torch.cat([rgb_features, dynamic_features], dim=1)
        fused = self.fusion_dropout(fused)
        return self.fusion_fc(fused)


class LateFusionResNetWithSPP(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5, pretrained: bool = False):
        super().__init__()
        self.rgb_branch = ResNetBranchWithSPP(pretrained=pretrained)
        self.dynamic_branch = ResNetBranchWithSPP(pretrained=pretrained)
        self.fusion_dropout = nn.Dropout(dropout_p)
        self.fusion_fc = nn.Linear(1024, num_classes)

    def forward(self, rgb_input: torch.Tensor, dynamic_input: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_branch(rgb_input)
        dyn_feat = self.dynamic_branch(dynamic_input)
        fused = torch.cat([rgb_feat, dyn_feat], dim=1)
        fused = self.fusion_dropout(fused)
        return self.fusion_fc(fused)


class LateFusionResNetAsymGridCBAM(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5, pretrained: bool = False):
        super().__init__()
        self.rgb_branch = ResNetBranchGridCBAM(pretrained=pretrained)
        self.dynamic_branch = ResNetBranchWithSPP(pretrained=pretrained)
        self.fusion_dropout = nn.Dropout(dropout_p)
        self.fusion_fc = nn.Linear(1024, num_classes)

    def forward(self, rgb_input: torch.Tensor, dynamic_input: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_branch(rgb_input)
        dyn_feat = self.dynamic_branch(dynamic_input)
        fused = torch.cat([rgb_feat, dyn_feat], dim=1)
        fused = self.fusion_dropout(fused)
        return self.fusion_fc(fused)


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class LateFusionResNetSE(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5, pretrained: bool = False):
        super().__init__()
        self.rgb_branch = ResNetBranchGridCBAM(pretrained=pretrained)
        self.dynamic_branch = ResNetBranchWithSPP(pretrained=pretrained)
        self.se_fusion = SEBlock(channel=1024, reduction=16)
        self.fusion_dropout = nn.Dropout(dropout_p)
        self.fusion_fc = nn.Linear(1024, num_classes)

    def forward(self, rgb_input: torch.Tensor, dynamic_input: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_branch(rgb_input)
        dyn_feat = self.dynamic_branch(dynamic_input)
        fused = torch.cat([rgb_feat, dyn_feat], dim=1)
        fused = self.se_fusion(fused)
        fused = self.fusion_dropout(fused)
        return self.fusion_fc(fused)


ABLATION_SPECS: dict[str, AblationSpec] = {
    "baseline_s": AblationSpec(
        key="baseline_s",
        display_name="Baseline-S",
        result_metrics_path=ROOT_DIR / "casme_rgb_train" / "rgb_result" / "result9" / "final_metrics.txt",
        branch_type="single_rgb",
        feature_strategy="ResNet18 + GAP",
        fusion_strategy="/",
        pretraining="ImageNet",
    ),
    "baseline_d": AblationSpec(
        key="baseline_d",
        display_name="Baseline-D",
        result_metrics_path=ROOT_DIR / "casme_dynamic_train" / "result1" / "final_metrics.txt",
        branch_type="single_dynamic",
        feature_strategy="ResNet18 + GAP",
        fusion_strategy="/",
        pretraining="ImageNet",
    ),
    "model_a": AblationSpec(
        key="model_a",
        display_name="Model A",
        result_metrics_path=ROOT_DIR / "casme2_fusion_train2_3class" / "fusion_result" / "final_metrics.txt",
        branch_type="dual",
        feature_strategy="RGB ResNet18(GAP) + Dynamic ResNet18(GAP)",
        fusion_strategy="Direct Concat",
        pretraining="ImageNet",
    ),
    "model_b": AblationSpec(
        key="model_b",
        display_name="Model B",
        result_metrics_path=ROOT_DIR / "casme2_fusion_train2_3class" / "fusion_result2_spp" / "final_metrics.txt",
        branch_type="dual",
        feature_strategy="RGB ResNet18(SPP) + Dynamic ResNet18(SPP)",
        fusion_strategy="Direct Concat",
        pretraining="ImageNet",
    ),
    "model_c": AblationSpec(
        key="model_c",
        display_name="Model C",
        result_metrics_path=ROOT_DIR / "casme2_fusion_train2_3class copy" / "fusion_result_asym_grid_cbam" / "final_metrics.txt",
        branch_type="dual",
        feature_strategy="RGB Grid+CBAM + Dynamic SPP",
        fusion_strategy="Direct Concat",
        pretraining="ImageNet",
    ),
    "adf_net": AblationSpec(
        key="adf_net",
        display_name="ADF-Net",
        result_metrics_path=ROOT_DIR / "casme2_fusion_train2_3class copy" / "fusion_result_asym_se-1" / "final_metrics.txt",
        branch_type="dual",
        feature_strategy="RGB Grid+CBAM + Dynamic SPP",
        fusion_strategy="SE Fusion",
        pretraining="ImageNet",
    ),
}


def list_model_keys() -> list[str]:
    return list(ABLATION_SPECS.keys())


def get_spec(model_key: str) -> AblationSpec:
    if model_key not in ABLATION_SPECS:
        raise KeyError(f"Unknown ablation model key: {model_key}")
    return ABLATION_SPECS[model_key]


def create_ablation_model(
    model_key: str,
    num_classes: int = 3,
    dropout_p: float = 0.5,
    pretrained: bool = False,
) -> nn.Module:
    if model_key == "baseline_s":
        return PlainSingleResNet18(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)
    if model_key == "baseline_d":
        return PlainSingleResNet18(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)
    if model_key == "model_a":
        return LateFusionResNetPlain(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)
    if model_key == "model_b":
        return LateFusionResNetWithSPP(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)
    if model_key == "model_c":
        return LateFusionResNetAsymGridCBAM(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)
    if model_key == "adf_net":
        return LateFusionResNetSE(num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained)
    raise KeyError(f"Unknown ablation model key: {model_key}")


def build_dummy_inputs(
    model_key: str,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    spec = get_spec(model_key)
    if spec.branch_type in {"single_rgb", "single_dynamic"}:
        return (torch.randn(batch_size, 3, height, width, device=device),)
    rgb = torch.randn(batch_size, 3, height, width, device=device)
    dynamic = torch.randn(batch_size, 3, height, width, device=device)
    return rgb, dynamic
