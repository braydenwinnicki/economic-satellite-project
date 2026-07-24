import sys  # allow adjusting import path for scripts executed as __main__
from pathlib import Path  # convenient path handling

# ensure project root is on sys.path so `models` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd  # read csv of image metadata
from models.caching import create_image_cache  # helper to build the cache
from models.resnet_multi import ResNetRegressor  # model used to get transforms


# read the processed CSV that lists all tiles and GEOIDs
df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts_tiles.csv",
    dtype={"GEOID": str},
)

# normalize column names (trim whitespace)
df.columns = df.columns.str.strip()


# instantiate the frozen resnet to reuse its preprocessing transforms
model = ResNetRegressor()
transform = model.weights.transforms()


# build and save the image cache for fast model training/evaluation
create_image_cache(
    df,
    transform,
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/census_images_cache.pt",
)
