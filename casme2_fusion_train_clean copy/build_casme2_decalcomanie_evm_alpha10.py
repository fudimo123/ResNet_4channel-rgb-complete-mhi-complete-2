import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(script_dir / "build_casme2_decalcomanie.py"),
        "--input-dir",
        str(script_dir / "casme2_aligned_rgb_evm_alpha10"),
        "--output-dir",
        str(script_dir / "casme2_static_decalcomanie_evm_alpha10"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
