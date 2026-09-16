import os
import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(script_dir / "build_samm_decalcomanie.py"),
        "--input-dir",
        str(script_dir / "samm_aligned_rgb_evm_classic_butter_o1"),
        "--output-dir",
        str(script_dir / "samm_static_decalcomanie_evm_classic_butter_o1"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
