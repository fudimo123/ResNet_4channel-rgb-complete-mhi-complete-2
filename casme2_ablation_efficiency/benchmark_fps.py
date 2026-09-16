from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from ablation_models import build_dummy_inputs, create_ablation_model, get_spec, list_model_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference FPS for CASME2 ablation models.")
    parser.add_argument("--models", nargs="*", default=list_model_keys(), choices=list_model_keys())
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-json", type=str, default="ablation_fps.json")
    parser.add_argument("--output-csv", type=str, default="ablation_fps.csv")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境未检测到 CUDA。")
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def measure_fps(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float, float]:
    for _ in range(warmup):
        _ = model(*inputs)
    synchronize_if_needed(device)

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(*inputs)
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start

    fps = iters * inputs[0].shape[0] / elapsed
    latency_ms = elapsed * 1000 / iters
    return fps, latency_ms


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
        fps, latency_ms = measure_fps(model, inputs, args.warmup, args.iters, device)
        rows.append(
            {
                "model_key": model_key,
                "display_name": spec.display_name,
                "branch_type": spec.branch_type,
                "feature_strategy": spec.feature_strategy,
                "fusion_strategy": spec.fusion_strategy,
                "pretraining": spec.pretraining,
                "device": str(device),
                "batch_size": args.batch_size,
                "input_height": args.height,
                "input_width": args.width,
                "warmup_iters": args.warmup,
                "measure_iters": args.iters,
                "fps": fps,
                "latency_ms": latency_ms,
            }
        )

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
                "device",
                "batch_size",
                "input_height",
                "input_width",
                "warmup_iters",
                "measure_iters",
                "fps",
                "latency_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("=" * 96)
    print("CASME2 Ablation FPS Benchmark")
    print("=" * 96)
    for row in rows:
        print(f"{row['display_name']:<12} | FPS: {row['fps']:>10.3f} | latency: {row['latency_ms']:>10.3f} ms")
    print(f"JSON saved to: {json_path}")
    print(f"CSV  saved to: {csv_path}")


if __name__ == "__main__":
    main()
