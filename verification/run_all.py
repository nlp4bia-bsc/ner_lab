"""Run every verification script and report the combined result."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = ("verify_data.py", "verify_encoding.py", "verify_encoder.py", "verify_models.py", "verify_training.py", "verify_evaluation.py", "verify_assessment.py", "verify_hpo.py")


def main() -> int:
    root = Path(__file__).parent
    failures = []

    for script in SCRIPTS:
        result = subprocess.run([sys.executable, str(root / script)])

        if result.returncode != 0:
            failures.append(script)

    print()

    if failures:
        print(f"FAILED: {', '.join(failures)}")

        return 1

    print(f"All {len(SCRIPTS)} verification scripts passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
