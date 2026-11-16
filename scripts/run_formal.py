"""Run the formal benchmark suite from the repository root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_eval.runner import main


if __name__ == "__main__":
    main()
