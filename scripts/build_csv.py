import geopandas as gpd
import pandas as pd

from src.satellite import get_image
from src.config import PROJECT_ROOT
from src.census import get_income_date

# get median_incomes from census api
income_df = get_income_date()

# get tracts for google maps api
# gpd.read_file loads a shapefile (a geospatial data format) into a GeoDataFrame
# Each row in the shapefile is one census tract with its boundary polygon
tracts = gpd.read_file(
    "/Users/braydenwinnicki/Downloads/cb_2025_09_tract_500k/cb_2025_09_tract_500k.shp"
)

csv_path = PROJECT_ROOT / "data" / "ct_tracts.csv"


# build dataset
rows = []

for i in range(len(tracts)):

    # .iloc[i] selects the i-th row by integer position from the GeoDataFrame
    current_tract = tracts.iloc[i]

    # .geometry.centroid computes the center point of the tract's boundary polygon
    # centroid.y is the latitude, centroid.x is the longitude
    centroid = current_tract.geometry.centroid

    lat = centroid.y
    lon = centroid.x
    # str() converts the GEOID number to a string so it matches the census data format
    geoid = str(current_tract["GEOID"])

    try:
        # get_image downloads a satellite image from Google Maps at (lat, lon)
        # and returns a dict with GEOID, lat, lon, and the saved file path
        rows.append(get_image(lat, lon, geoid))
    except Exception as e:
        print(f"Failed at {i}: {e}")

    # save every 100 file for safety, greater than 0 checks for a none empty datset
    # % is the modulo operator — len(rows) % 100 == 0 means "every 100 rows"
    if len(rows) > 0 and len(rows) % 100 == 0:
        # pd.DataFrame(rows) converts the list of dicts into a table
        image_df = pd.DataFrame(rows)

        # merge with income data
        # .merge() joins two DataFrames on a common column ("GEOID")
        # how="left" means keep all rows from the left DataFrame (images)
        # even if there's no matching income (would get NaN for income)
        checkpoint_df = image_df.merge(income_df, on="GEOID", how="left")
        # index=False means don't write row numbers to the CSV
        checkpoint_df.to_csv(csv_path, index=False)
        print(f"Checkpoint saved ({len(rows)} rows)")

# final save
image_df = pd.DataFrame(rows)

# Same merge as above but for the complete dataset
final_df = image_df.merge(income_df, on="GEOID", how="left")

final_df.to_csv(csv_path, index=False)

print("****Full Dataset saved**** →", csv_path)