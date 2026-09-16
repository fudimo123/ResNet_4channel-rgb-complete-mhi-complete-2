from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ablation_models import build_dummy_inputs, create_ablation_model, get_spec, list_model_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile FLOPs and Params for CASME2 ablation models.")
    parser.add_argument("--models", nargs="*", default=list_model_keys(), choices=list_model_keys())
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-json", type=str, default="ablation_flops_params.json")
    parser.add_argument("--output-csv", type=str, default="ablation_flops_params.csv")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境未检测到 CUDA。")
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def try_thop(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[int | None, str]:
    try:
        from thop import profile

        flops, _ = profile(model, inputs=inputs, verbose=False)
        return int(flops), "thop"
    except Exception:
        return None, ""


def try_fvcore(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[int | None, str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, inputs).total()
        return int(flops), "fvcore"
    except Exception:
        return None, ""


def measure_flops(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[int | None, str]:
    flops, backend = try_thop(model, inputs)
    if flops is not None:
        return flops, backend
    flops, backend = try_fvcore(model, inputs)
    if flops is not None:
        return flops, backend
    return None, "unavailable"


def pretty_count(value: int | None) -> str:
    if value is None:
        return "N/A"
    units = ["", "K", "M", "G", "T"]
    scaled = float(value)
    idx = 0
    while scaled >= 1000 and idx < len(units) - 1:
        scaled /= 1000
        idx += 1
    return f"{scaled:.3f} {units[idx]}".strip()


def pretty_flops(value: int | None) -> str:
    if value is None:
        return "N/A"
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs"]
    scaled = float(value)
    idx = 0
    while scaled >= 1000 and idx < len(units) - 1:
        scaled /= 1000
        idx += 1
    return f"{scaled:.3f} {units[idx]}"


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    device = resolve_device(args.device)

    rows = []
    for model_key in args.models:
        spec = get_spec(model_key)
        model = create_ablation_model(model_key=model_key, num_classes=3, dropout_p=0.5, pretrained=False).to(device)
        model.eval()
        inputs = build_dummy_inputs(model_key, args.batch_size, args.height, args.width, device)
        total_params, trainable_params = count_parameters(model)
        flops, flops_backend = measure_flops(model, inputs)

        row = {
            "model_key": model_key,
            "display_name": spec.display_name,
            "metrics_path": str(spec.result_metrics_path),
            "branch_type": spec.branch_type,
            "feature_strategy": spec.feature_strategy,
            "fusion_strategy": spec.fusion_strategy,
            "pretraining": spec.pretraining,
            "input_shape": [args.batch_size, 3, args.height, args.width],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": flops,
            "flops_backend": flops_backend,
        }
        rows.append(row)

    json_path = (script_dir / args.output_json).resolve()
    csv_path = (script_dir / args.output_csv).resolve()
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_key",
                "display_name",
                "branch_type",
                "feature_strategy",
                "fusion_strategy",
                "pretraining",
                "input_shape",
                "total_params",
                "trainable_params",
                "flops",
                "flops_backend",
                "metrics_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("=" * 96)
    print("CASME2 Ablation Params/FLOPs")
    print("=" * 96)
    for row in rows:
        print(
            f"{row['display_name']:<12} | Params: {pretty_count(row['total_params']):>12} | "
            f"FLOPs: {pretty_flops(row['flops']):>16} | backend: {row['flops_backend']}"
        )
    print(f"JSON saved to: {json_path}")
    print(f"CSV  saved to: {csv_path}")


if __name__ == "__main__":
    main()
