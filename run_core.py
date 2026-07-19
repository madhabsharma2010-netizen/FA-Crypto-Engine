"""Launcher that runs from the project root so package imports work."""

import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "main.py"), run_name="__main__")
