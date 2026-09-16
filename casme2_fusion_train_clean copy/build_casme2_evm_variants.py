import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

SIMPLE_VARIANTS = [
    (
        "alpha6",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_alpha6"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_alpha6"),
            "--alpha", "6",
        ],
    ),
    (
        "alpha10",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_alpha10"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_alpha10"),
            "--alpha", "10",
        ],
    ),
    (
        "alpha12",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_alpha12"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_alpha12"),
            "--alpha", "12",
        ],
    ),
    (
        "fh25",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_fh25"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_fh25"),
            "--freq-high", "2.5",
        ],
    ),
    (
        "fh35",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_fh35"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_fh35"),
            "--freq-high", "3.5",
        ],
    ),
    (
        "fh40",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_fh40"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_fh40"),
            "--freq-high", "4.0",
        ],
    ),
    (
        "pyr2",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_pyr2"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_pyr2"),
            "--pyramid-level", "2",
        ],
    ),
    (
        "alpha10_fh35",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_alpha10_fh35"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_alpha10_fh35"),
            "--alpha", "10",
            "--freq-high", "3.5",
        ],
    ),
]

CLASSIC_VARIANTS = [
    (
        "classic_butter_o1",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_classic_butter_o1"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_classic_butter_o1"),
            "--filter-type", "butterworth",
            "--butter-order", "1",
        ],
    ),
    (
        "classic_butter_o2",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_classic_butter_o2"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_classic_butter_o2"),
            "--filter-type", "butterworth",
            "--butter-order", "2",
        ],
    ),
    (
        "classic_fft_ideal",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "casme2_aligned_rgb_evm_classic_fft_ideal"),
            "--output-dyn-dir", str(SCRIPT_DIR / "casme2_dynamic_data_evm_classic_fft_ideal"),
            "--filter-type", "ideal",
            "--butter-order", "1",
        ],
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build all CASME2 EVM / classic EVM preprocessing variants."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device passed to the underlying MTCNN-based preprocessing scripts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing outputs for every variant.",
    )
    return parser.parse_args()


def run_variant(label, base_script, extra_args, shared_args):
    cmd = [sys.executable, str(SCRIPT_DIR / base_script), *extra_args, *shared_args]
    print(f"\n=== Building {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    shared_args = ["--device", args.device]
    if args.overwrite:
        shared_args.append("--overwrite")

    summary = {"simple": [], "classic": []}

    for label, args in SIMPLE_VARIANTS:
        run_variant(label, "preprocess_casme2_evm.py", args, shared_args)
        summary["simple"].append(label)

    for label, args in CLASSIC_VARIANTS:
        run_variant(label, "preprocess_casme2_evm_classic.py", args, shared_args)
        summary["classic"].append(label)

    summary_path = SCRIPT_DIR / "casme2_evm_variant_matrix.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved variant index to: {summary_path}")


if __name__ == "__main__":
    main()
