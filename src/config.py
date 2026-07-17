from pathlib import Path

# Paths (stable project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"  # top-level data directory
RAW_DIR = DATA_DIR / "raw"  # raw/unprocessed data
IMAGE_DIR = RAW_DIR / "images"  # where downloaded satellite images are saved

# ensure the image directory exists
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
