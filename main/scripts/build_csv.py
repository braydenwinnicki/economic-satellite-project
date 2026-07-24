import geopandas as gpd
import pandas as pd

from src.satellite import get_image
from src.config import PROJECT_ROOT
from src.census import get_income_date

# get median_incomes from census api
income_df = get_income_date()

# get tracts for google maps api
tracts = gpd.read_file(
    "/Users/braydenwinnicki/Downloads/cb_2025_09_tract_500k/cb_2025_09_tract_500k.shp"
)

csv_path = PROJECT_ROOT / "data" / "ct_tracts.csv"


# build dataset
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

    # save every 100 file for safety, greater than 0 checks for a none empty datset
    if len(rows) > 0 and len(rows) % 100 == 0:
        image_df = pd.DataFrame(rows)

        # merge with income data
        checkpoint_df = image_df.merge(income_df, on="GEOID", how="left")
        checkpoint_df.to_csv(csv_path, index=False)
        print(f"Checkpoint saved ({len(rows)} rows)")

# final save
image_df = pd.DataFrame(rows)

final_df = image_df.merge(income_df, on="GEOID", how="left")

final_df.to_csv(csv_path, index=False)

print("****Full Dataset saved**** →", csv_path)
