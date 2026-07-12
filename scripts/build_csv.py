import geopandas as gpd
import pandas as pd
import numpy as np

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

# reproject to a meters-based CRS so area is in real units, not degrees
tracts = tracts.to_crs(epsg=3857)

MIN_TILES = 4
MAX_TILES = 25


def tiles_for_area(area_sq_m):
    """Scale tile count with the square root of area (so tile count
    scales with linear dimension, not raw area - keeps things from
    exploding for huge rural tracts)."""
    area_sq_km = area_sq_m / 1_000_000 #convert to km for ease 
    n = int(np.sqrt(area_sq_km)) + MIN_TILES # use sqrt to focus on linear side lengths so crazy areas dont mess things up
    return max(MIN_TILES, min(MAX_TILES, n)) # if n is in range return it, if not send back other option


def get_grid_points(polygon, n):
    """Return up to n points spread across the polygon's bounding box,
    keeping only ones that fall inside the actual tract shape."""
    minx, miny, maxx, maxy = polygon.bounds #get x,y of smallest rectangle that can fully fit the tract 
    side = int(np.ceil(np.sqrt(n)))  # we need to make n points arranged a sgird so use square root to grab that, round up

    xs = np.linspace(minx, maxx, side + 2)[1:-1] # generates side # of a evenly spaced points on x axis. cuts off forst and alst becuase they are unlikley to be in the actual tract
    ys = np.linspace(miny, maxy, side + 2)[1:-1] # does same for y


points = []

for x in xs: # loop over each point on x axis
    for y in ys: # check each y value, creating a grid
        pt = gpd.points_from_xy([x], [y])[0] # convert to gpd spatial point for checks 
        if polygon.contains(pt): # confirms the spacial point is inside the actually tract
            points.append((y, x)) # save the point 

if len(points) == 0: # if its zero, give us at least one photo 
    c = polygon.centroid
    points = [(c.y, c.x)]

return points[:n] # trim down in case grid made it go over


# build dataset

# keep original (lat/lon) geometry around for get_image, since we reprojected above
tracts_latlon = gpd.read_file(
    "/Users/braydenwinnicki/Downloads/cb_2025_09_tract_500k/cb_2025_09_tract_500k.shp"
)

for i in range(len(tracts)):  #grab each tract
    geoid = str(tracts.iloc[i]["GEOID"]) #get its GEOID
    area = tracts.iloc[i].geometry.area  # in sq meters now, thanks to reprojection
    n_tiles = tiles_for_area(area) # get amount of pics needed for tract

    # sample points using the lat/lon version (Google Maps wants lat/lon, not meters)
    polygon_latlon = tracts_latlon.iloc[i].geometry # grab this tracts lat/lon
    points = get_grid_points(polygon_latlon, n_tiles) # create the grid points for pic locations

    for tile_idx, (lat, lon) in enumerate(points): # for each tile(with locations) 
        try:
            result = get_image(lat, lon, f"{geoid}_{tile_idx}") # get_image 
            result["GEOID"] = geoid
            result["tile_idx"] = tile_idx
            result["n_tiles_total"] = len(points)
            rows.append(result)
        except Exception as e:
            print(f"Failed at tract {i} ({geoid}), tile {tile_idx}: {e}")

    if len(rows) > 0 and len(rows) % 500 == 0:
        image_df = pd.DataFrame(rows)
        checkpoint_df = image_df.merge(income_df, on="GEOID", how="left")
        checkpoint_df.to_csv(csv_path, index=False)
        print(f"Checkpoint saved ({len(rows)} rows)")

image_df = pd.DataFrame(rows)
final_df = image_df.merge(income_df, on="GEOID", how="left")
final_df.to_csv(csv_path, index=False)
print("****Full Dataset saved**** →", csv_path)
