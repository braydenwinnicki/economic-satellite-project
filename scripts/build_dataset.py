import geopandas as gpd
import pandas as pd

from src.satellite import get_image
from src.config import PROJECT_ROOT

#get tracts
tracts = gpd.read_file(
    "/Users/braydenwinnicki/Downloads/cb_2025_09_tract_500k/cb_2025_09_tract_500k.shp"
)

csv_path = PROJECT_ROOT / "data" / "ct_tracts.csv"


#build dataset
rows = []

for i in range(len(tracts)):

    current_tract = tracts.iloc[i]

    centroid = current_tract.geometry.centroid

    lat = centroid.y
    lon = centroid.x
    geoid = str(current_tract["GEOID"])

    try:
        rows.append(get_image(lat, lon, geoid))
    except Exception as e:
        print(f"Failed at {i}: {e}")
    
    if len(rows) % 10 == 0:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Checkpoint saved ({len(rows)} rows)")


df = pd.DataFrame(rows)

df.to_csv(csv_path, index=False)

print("Dataset saved →", csv_path)
