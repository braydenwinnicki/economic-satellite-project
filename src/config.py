from pathlib import Path  # cross-platform path handling (works on Windows/Mac/Linux)

# Paths (stable project root)
# __file__ is this file's path, .resolve() makes it absolute, .parents[1] goes up 2 levels
# (src/ -> econ_project/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"  # top-level data directory
RAW_DIR = DATA_DIR / "raw"  # raw/unprocessed data
IMAGE_DIR = RAW_DIR / "images"  # where downloaded satellite images are saved

# ensure the image directory exists
# parents=True creates any missing parent dirs (data/ and raw/ if they don't exist)
# exist_ok=True means it won't error if the dir already exists
IMAGE_DIR.mkdir(parents=True, exist_ok=True)