"""Repository entry point; works without an editable install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pmse_benchmark.cli import main

if __name__ == "__main__":
    main()
