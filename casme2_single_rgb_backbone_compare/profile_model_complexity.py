from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fusion_model_se import create_fusion_model_se


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("fusion_result_asym_se-1") / "fusion_resnet18_casme2_final_model_for_gui.pth"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("fusion_result_asym_se-1") / "model_complexity_report.json"),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境未检测到 CUDA。")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(script_dir: Path, checkpoint_rel: str, device: torch.device) -> tuple[torch.nn.Module, Path]:
    checkpoint_path = (script_dir / checkpoint_rel).resolve()
    model = create_fusion_model_se(num_classes=3, dropout_p=0.5, pretrained=False, transfer_weights_path=None)
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, checkpoint_path


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def build_inputs(batch_size: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = torch.randn(batch_size, 3, height, width, device=device)
    dynamic = torch.randn(batch_size, 3, height, width, device=device)
    return rgb, dynamic


def try_thop(model: torch.nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[int | None, str]:
    try:
        from thop import profile

        flops, _ = profile(model, inputs=inputs, verbose=False)
        return int(flops), "thop"
    except Exception:
        return None, ""


def try_fvcore(model: torch.nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[int | None, str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, inputs).total()
        return int(flops), "fvcore"
    except Exception:
        return None, ""


def measure_flops(model: torch.nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[int | None, str]:
    flops, backend = try_thop(model, inputs)
    if flops is not None:
        return flops, backend
    flops, backend = try_fvcore(model, inputs)
    if flops is not None:
        return flops, backend
    return None, ""


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def measure_fps(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
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


def format_count(value: int | None) -> str:
    if value is None:
        return "N/A"
    units = ["", "K", "M", "G", "T"]
    scaled = float(value)
    unit_index = 0
    while scaled >= 1000 and unit_index < len(units) - 1:
        scaled /= 1000
        unit_index += 1
    return f"{scaled:.3f} {units[unit_index]}".strip()


def format_flops(value: int | None) -> str:
    if value is None:
        return "N/A"
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs"]
    scaled = float(value)
    unit_index = 0
    while scaled >= 1000 and unit_index < len(units) - 1:
        scaled /= 1000
        unit_index += 1
    return f"{scaled:.3f} {units[unit_index]}"


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    device = resolve_device(args.device)
    model, checkpoint_path = load_model(script_dir, args.checkpoint, device)
    inputs = build_inputs(args.batch_size, args.height, args.width, device)

    total_params, trainable_params = count_parameters(model)
    flops, flops_backend = measure_flops(model, inputs)
    fps, latency_ms = measure_fps(model, inputs, args.warmup, args.iters, device)

    result = {
        "model_name": "LateFusionResNetSE",
        "experiment_dir": "fusion_result_asym_se-1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "input_shape_per_branch": [args.batch_size, 3, args.height, args.width],
        "device": str(device),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "flops_backend": flops_backend if flops is not None else "unavailable",
        "fps": fps,
        "latency_ms": latency_ms,
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
    }

    print("=" * 72)
    print("ADF-Net Complexity Profile")
    print("=" * 72)
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Checkpoint found : {checkpoint_path.exists()}")
    print(f"Device           : {device}")
    print(f"Input per branch : {tuple(result['input_shape_per_branch'])}")
    print(f"Total Params     : {total_params} ({format_count(total_params)})")
    print(f"Trainable Params : {trainable_params} ({format_count(trainable_params)})")
    if flops is None:
        print("FLOPs            : N/A")
        print("FLOPs backend    : unavailable (可安装 thop 或 fvcore 后重新运行)")
    else:
        print(f"FLOPs            : {flops} ({format_flops(flops)})")
        print(f"FLOPs backend    : {flops_backend}")
    print(f"FPS              : {fps:.3f}")
    print(f"Latency          : {latency_ms:.3f} ms / iteration")

    output_path = (script_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to  : {output_path}")


if __name__ == "__main__":
    main()
