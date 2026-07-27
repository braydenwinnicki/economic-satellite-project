from pathlib import Path

# Paths (stable project root)

# Path(__file__) = config.py's path
# .resolve() makes it absolute (resolves symlinks, relative parts like "..")
# .parents[1] = 2 directories up from config.py, which is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# / operator concatenates path components (unlike os.path.join)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = RAW_DIR / "images"

# .mkdir(parents=True, exist_ok=True) creates the directory and any missing parent directories
# parents=True means it creates data/raw/ if those don't exist yet
# exist_ok=True means it won't error if the directory already exists
IMAGE_DIR.mkdir(parents=True, exist_ok=True)