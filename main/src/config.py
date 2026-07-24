from pathlib import Path

# Paths (stable project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = RAW_DIR / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
