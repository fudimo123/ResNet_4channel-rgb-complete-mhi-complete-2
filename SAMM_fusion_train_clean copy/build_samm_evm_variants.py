import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

SIMPLE_VARIANTS = [
    (
        "alpha6",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_alpha6"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_alpha6"),
            "--alpha", "6",
        ],
    ),
    (
        "alpha10",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_alpha10"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_alpha10"),
            "--alpha", "10",
        ],
    ),
    (
        "alpha12",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_alpha12"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_alpha12"),
            "--alpha", "12",
        ],
    ),
    (
        "fh25",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_fh25"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_fh25"),
            "--freq-high", "2.5",
        ],
    ),
    (
        "fh35",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_fh35"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_fh35"),
            "--freq-high", "3.5",
        ],
    ),
    (
        "fh40",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_fh40"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_fh40"),
            "--freq-high", "4.0",
        ],
    ),
    (
        "pyr2",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_pyr2"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_pyr2"),
            "--pyramid-level", "2",
        ],
    ),
    (
        "alpha10_fh35",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_alpha10_fh35"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_alpha10_fh35"),
            "--alpha", "10",
            "--freq-high", "3.5",
        ],
    ),
]

CLASSIC_VARIANTS = [
    (
        "classic_butter_o1",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_classic_butter_o1"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_classic_butter_o1"),
            "--filter-type", "butterworth",
            "--butter-order", "1",
        ],
    ),
    (
        "classic_butter_o2",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_classic_butter_o2"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_classic_butter_o2"),
            "--filter-type", "butterworth",
            "--butter-order", "2",
        ],
    ),
    (
        "classic_fft_ideal",
        [
            "--output-rgb-dir", str(SCRIPT_DIR / "samm_aligned_rgb_evm_classic_fft_ideal"),
            "--output-dyn-dir", str(SCRIPT_DIR / "samm_dynamic_data_evm_classic_fft_ideal"),
            "--filter-type", "ideal",
            "--butter-order", "1",
        ],
    ),
]


def run_variant(label, base_script, extra_args):
    cmd = [sys.executable, str(SCRIPT_DIR / base_script), *extra_args]
    print(f"\n=== Building {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    summary = {"simple": [], "classic": []}

    for label, args in SIMPLE_VARIANTS:
        run_variant(label, "preprocess_samm_evm.py", args)
        summary["simple"].append(label)

    for label, args in CLASSIC_VARIANTS:
        run_variant(label, "preprocess_samm_evm_classic.py", args)
        summary["classic"].append(label)

    summary_path = SCRIPT_DIR / "samm_evm_variant_matrix.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved variant index to: {summary_path}")


if __name__ == "__main__":
    main()
