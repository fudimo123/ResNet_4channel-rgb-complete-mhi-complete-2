from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT_DIR / "casme2_fusion_train2_3class copy"
CHECKPOINT_PATH = EXPERIMENT_DIR / "fusion_result_asym_se-1" / "fusion_resnet18_casme2_final_model_for_gui.pth"
ANNOTATION_PATH = ROOT_DIR / "data" / "CASME2_RAW_selected" / "CASME2-coding-20140508.xlsx"
RAW_ROOT = ROOT_DIR / "data" / "CASME2_RAW_selected" / "CASME2_RAW_selected"
DYNAMIC_ROOT = EXPERIMENT_DIR / "dynamic_data"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
MARGIN = 20

EMOTION_MAP = {
    "happiness": "positive",
    "disgust": "negative",
    "repression": "negative",
    "sadness": "negative",
    "fear": "negative",
    "surprise": "surprise",
}
CLASS_NAMES = ["positive", "negative", "surprise"]

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from fusion_model_se import create_fusion_model_se  # noqa: E402


@dataclass
class SampleRecord:
    subject_num: int
    subject_id: str
    sequence: str
    raw_emotion: str
    emotion: str
    onset: int
    apex: int | None
    offset: int
    rgb_frame_path: Path
    dynamic_image_path: Path


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module) -> None:
        self.model = model
        self.target_module = target_module
        self.activations: torch.Tensor | None = None
        self.forward_handle = self.target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

    def remove(self) -> None:
        self.forward_handle.remove()

    def generate(
        self,
        rgb_tensor: torch.Tensor,
        dynamic_tensor: torch.Tensor,
        target_class: int | None = None,
        method: str = "gradcam",
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        self.activations = None

        logits = self.model(rgb_tensor, dynamic_tensor)
        predicted_class = int(torch.argmax(logits, dim=1).item())
        if target_class is None:
            target_class = predicted_class

        score = logits[:, target_class].sum()
        if self.activations is None:
            raise RuntimeError("Grad-CAM hook did not capture activations.")

        gradients = torch.autograd.grad(score, self.activations, retain_graph=True)[0]

        if method == "gradcam++":
            grads_sq = gradients.pow(2)
            grads_cu = grads_sq * gradients
            sum_activations = self.activations.sum(dim=(2, 3), keepdim=True)
            eps = 1e-8
            alpha = grads_sq / (2.0 * grads_sq + sum_activations * grads_cu + eps)
            alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))
            weights = (torch.relu(gradients) * alpha).sum(dim=(2, 3), keepdim=True)
        else:
            weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).detach().cpu().numpy()
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam, predicted_class, logits.detach().cpu().squeeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ADF-Net Grad-CAM samples for paper visualization.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument(
        "--rgb-target",
        default="cbam",
        choices=["cbam", "layer4", "layer3"],
        help="Target layer used to generate CAM for the RGB branch.",
    )
    parser.add_argument(
        "--dynamic-target",
        default="layer4",
        choices=["layer4", "layer3", "layer2"],
        help="Target layer used to generate CAM for the dynamic branch.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "gradcampp"],
        choices=["gradcam", "gradcampp"],
        help="Visualization methods to generate.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = create_fusion_model_se(num_classes=3, dropout_p=0.5, pretrained=False, transfer_weights_path=None)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def get_mtcnn(device: torch.device) -> MTCNN:
    mtcnn_device = "cuda" if device.type == "cuda" else "cpu"
    return MTCNN(keep_all=False, device=mtcnn_device)


def find_existing_frame(sequence_dir: Path, preferred_frame: int | None, onset: int, offset: int) -> Path:
    frame_candidates = sorted(sequence_dir.glob("img*.jpg"))
    if not frame_candidates:
        raise FileNotFoundError(f"No frames found under {sequence_dir}")

    frame_map: dict[int, Path] = {}
    for frame_path in frame_candidates:
        name = frame_path.stem
        try:
            frame_num = int(name[3:])
        except ValueError:
            continue
        frame_map[frame_num] = frame_path

    if preferred_frame is not None and preferred_frame in frame_map:
        return frame_map[preferred_frame]

    valid = [(num, path) for num, path in frame_map.items() if onset <= num <= offset]
    if valid:
        target = preferred_frame if preferred_frame is not None else (onset + offset) // 2
        best_num, best_path = min(valid, key=lambda item: abs(item[0] - target))
        return best_path

    return frame_candidates[len(frame_candidates) // 2]


def select_samples(samples_per_class: int) -> list[SampleRecord]:
    df = pd.read_excel(ANNOTATION_PATH)
    selected: list[SampleRecord] = []
    counts = {name: 0 for name in CLASS_NAMES}

    for _, row in df.iterrows():
        subject_num = int(row["Subject"])
        if subject_num == 18:
            continue

        raw_emotion = str(row["Estimated Emotion"]).strip().lower()
        mapped = EMOTION_MAP.get(raw_emotion)
        if mapped is None or counts[mapped] >= samples_per_class:
            continue

        sequence = str(row["Filename"])
        subject_id = f"sub{subject_num:02d}"
        sequence_dir = RAW_ROOT / subject_id / sequence
        if not sequence_dir.exists():
            continue

        onset = int(row["OnsetFrame"])
        offset = int(row["OffsetFrame"])
        apex_raw = row["ApexFrame"]
        apex = int(apex_raw) if pd.notna(apex_raw) else None
        dynamic_path = DYNAMIC_ROOT / subject_id / f"s{subject_num}_{sequence}.jpg"
        if not dynamic_path.exists():
            continue

        rgb_frame_path = find_existing_frame(sequence_dir, apex, onset, offset)
        selected.append(
            SampleRecord(
                subject_num=subject_num,
                subject_id=subject_id,
                sequence=sequence,
                raw_emotion=raw_emotion,
                emotion=mapped,
                onset=onset,
                apex=apex,
                offset=offset,
                rgb_frame_path=rgb_frame_path,
                dynamic_image_path=dynamic_path,
            )
        )
        counts[mapped] += 1

        if all(count >= samples_per_class for count in counts.values()):
            break

    missing = [name for name, count in counts.items() if count < samples_per_class]
    if missing:
        raise RuntimeError(f"Could not find enough samples for classes: {missing}")
    return selected


def detect_face_box(mtcnn: MTCNN, image: Image.Image) -> tuple[int, int, int, int] | None:
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
    x1 = max(0, x1 - MARGIN)
    y1 = max(0, y1 - MARGIN)
    x2 = min(image.width, x2 + MARGIN)
    y2 = min(image.height, y2 + MARGIN)
    return x1, y1, x2, y2


def crop_face(image: Image.Image, mtcnn: MTCNN) -> Image.Image:
    box = detect_face_box(mtcnn, image)
    if box is None:
        return image
    return image.crop(box)


def get_transform() -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def to_display_array(image: Image.Image) -> np.ndarray:
    return np.array(image.resize((224, 224)).convert("RGB"))


def overlay_cam_on_image(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = np.uint8(np.clip(cam, 0, 1) * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip((1 - alpha) * image_rgb + alpha * heatmap, 0, 255).astype(np.uint8)
    return overlay


def save_single_branch_figure(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    overlay = overlay_cam_on_image(image_rgb, cam)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dual_branch_panel(
    rgb_image: np.ndarray,
    dynamic_image: np.ndarray,
    rgb_cam: np.ndarray,
    dynamic_cam: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    rgb_overlay = overlay_cam_on_image(rgb_image, rgb_cam)
    dynamic_overlay = overlay_cam_on_image(dynamic_image, dynamic_cam)
    combined_cam = np.clip((rgb_cam + dynamic_cam) / 2.0, 0.0, 1.0)
    combined_overlay = overlay_cam_on_image(rgb_image, combined_cam)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("RGB Input")
    axes[0, 1].imshow(rgb_cam, cmap="jet")
    axes[0, 1].set_title("RGB Branch CAM")
    axes[0, 2].imshow(rgb_overlay)
    axes[0, 2].set_title("RGB Overlay")
    axes[1, 0].imshow(dynamic_image)
    axes[1, 0].set_title("Dynamic Input")
    axes[1, 1].imshow(dynamic_cam, cmap="jet")
    axes[1, 1].set_title("Dynamic Branch CAM")
    axes[1, 2].imshow(combined_overlay)
    axes[1, 2].set_title("Dual-Branch Overlay")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def method_label(method: str) -> str:
    return "Grad-CAM++" if method == "gradcampp" else "Grad-CAM"


def resolve_target_module(branch: torch.nn.Module, target_name: str) -> torch.nn.Module:
    if not hasattr(branch, target_name):
        raise ValueError(f"Target layer '{target_name}' not found in branch {branch.__class__.__name__}.")
    return getattr(branch, target_name)


def softmax_probs(logits: torch.Tensor) -> list[float]:
    probs = torch.softmax(logits, dim=0).cpu().numpy().tolist()
    return [float(v) for v in probs]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, device)
    transform = get_transform()
    mtcnn = get_mtcnn(device)
    samples = select_samples(args.samples_per_class)

    rgb_target_module = resolve_target_module(model.rgb_branch, args.rgb_target)
    dynamic_target_module = resolve_target_module(model.dynamic_branch, args.dynamic_target)
    rgb_cam_runner = GradCAM(model, rgb_target_module)
    dynamic_cam_runner = GradCAM(model, dynamic_target_module)

    results = []
    for sample in samples:
        rgb_image = Image.open(sample.rgb_frame_path).convert("RGB")
        dynamic_image = Image.open(sample.dynamic_image_path).convert("RGB")

        rgb_face = crop_face(rgb_image, mtcnn)
        dynamic_face = crop_face(dynamic_image, mtcnn)

        rgb_display = to_display_array(rgb_face)
        dynamic_display = to_display_array(dynamic_face)

        rgb_tensor = transform(rgb_face).unsqueeze(0).to(device)
        dynamic_tensor = transform(dynamic_face).unsqueeze(0).to(device)

        sample_name = f"{sample.subject_id}_{sample.sequence}_{sample.emotion}"
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        rgb_input_path = sample_dir / "rgb_input.png"
        dynamic_input_path = sample_dir / "dynamic_input.png"
        Image.fromarray(rgb_display).save(rgb_input_path)
        Image.fromarray(dynamic_display).save(dynamic_input_path)

        predicted_class: int | None = None
        logits: torch.Tensor | None = None
        method_outputs: dict[str, dict[str, str]] = {}
        for method in args.methods:
            internal_method = "gradcam++" if method == "gradcampp" else "gradcam"
            rgb_cam, current_predicted_class, current_logits = rgb_cam_runner.generate(
                rgb_tensor,
                dynamic_tensor,
                method=internal_method,
            )
            if predicted_class is None:
                predicted_class = current_predicted_class
                logits = current_logits
            dynamic_cam, _, _ = dynamic_cam_runner.generate(
                rgb_tensor,
                dynamic_tensor,
                target_class=predicted_class,
                method=internal_method,
            )

            rgb_cam = cv2.resize(rgb_cam, (224, 224), interpolation=cv2.INTER_LINEAR)
            dynamic_cam = cv2.resize(dynamic_cam, (224, 224), interpolation=cv2.INTER_LINEAR)

            rgb_single_path = sample_dir / f"single_branch_rgb_{method}.png"
            dynamic_single_path = sample_dir / f"single_branch_dynamic_{method}.png"
            dual_panel_path = sample_dir / f"dual_branch_{method}_panel.png"

            label = method_label(method)
            save_single_branch_figure(rgb_display, rgb_cam, f"RGB Branch {label}", rgb_single_path)
            save_single_branch_figure(dynamic_display, dynamic_cam, f"Dynamic Branch {label}", dynamic_single_path)
            save_dual_branch_panel(
                rgb_display,
                dynamic_display,
                rgb_cam,
                dynamic_cam,
                f"{sample.subject_id} / {sample.sequence} / GT={sample.emotion} / Pred={CLASS_NAMES[predicted_class]} / {label}",
                dual_panel_path,
            )
            method_outputs[method] = {
                "single_branch_rgb_path": str(rgb_single_path),
                "single_branch_dynamic_path": str(dynamic_single_path),
                "dual_branch_panel_path": str(dual_panel_path),
            }

        if predicted_class is None or logits is None:
            raise RuntimeError("No CAM methods were generated.")

        record = {
            "sample_name": sample_name,
            "subject_num": sample.subject_num,
            "subject_id": sample.subject_id,
            "sequence": sample.sequence,
            "raw_emotion": sample.raw_emotion,
            "mapped_emotion": sample.emotion,
            "predicted_class": CLASS_NAMES[predicted_class],
            "probabilities": dict(zip(CLASS_NAMES, softmax_probs(logits))),
            "rgb_frame_path": str(sample.rgb_frame_path),
            "dynamic_image_path": str(sample.dynamic_image_path),
            "rgb_input_path": str(rgb_input_path),
            "dynamic_input_path": str(dynamic_input_path),
            "rgb_target": args.rgb_target,
            "dynamic_target": args.dynamic_target,
            "methods": method_outputs,
        }
        (sample_dir / "summary.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(record)

    rgb_cam_runner.remove()
    dynamic_cam_runner.remove()

    index = {
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "rgb_target": args.rgb_target,
        "dynamic_target": args.dynamic_target,
        "num_samples": len(results),
        "samples": results,
    }
    (output_dir / "gradcam_results_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
