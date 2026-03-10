#!/usr/bin/env python3
"""Train all fire detection models (MLP + TabPFN classification + TabPFN regression).

Calls train-all-models.sh as a subprocess. Designed for use in Databricks.
"""

import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    shell_script = script_dir / "train-all-models.sh"

    if not shell_script.exists():
        print(f"ERROR: {shell_script} not found", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        ["bash", str(shell_script)],
        cwd=str(script_dir),
        text=True,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
