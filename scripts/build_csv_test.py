import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
import numpy as np

from src.satellite import get_image
from src.config import PROJECT_ROOT
from src.census import get_income_date

income_df = get_income_date()

tracts = gpd.read_file(
    "/Users/braydenwinnicki/Downloads/cb_2025_09_tract_500k/cb_2025_09_tract_500k.shp"
)
tracts = tracts.to_crs(epsg=3857)

tracts_latlon = gpd.read_file(
    "/Users/braydenwinnicki/Downloads/cb_2025_09_tract_500k/cb_2025_09_tract_500k.shp"
)

# TEST MODE: only process a handful of tracts before committing to the full run
N_TEST_TRACTS = 15
tracts = tracts.iloc[:N_TEST_TRACTS]
tracts_latlon = tracts_latlon.iloc[:N_TEST_TRACTS]

csv_path = PROJECT_ROOT / "data" / "ct_tracts_tiles_TEST.csv"

MIN_TILES = 4
MAX_TILES = 50
MULTIPLIER = 3
ZOOM = 17


def tiles_for_area(area_sq_m):
    area_sq_km = area_sq_m / 1_000_000
    n = int(np.sqrt(area_sq_km) * MULTIPLIER) + MIN_TILES
    return max(MIN_TILES, min(MAX_TILES, n))


def get_grid_points(polygon, n):
    minx, miny, maxx, maxy = polygon.bounds
    side = int(np.ceil(np.sqrt(n)))

    xs = np.linspace(minx, maxx, side + 2)[1:-1]
    ys = np.linspace(miny, maxy, side + 2)[1:-1]

    points = []
    for x in xs:
        for y in ys:
            pt = gpd.points_from_xy([x], [y])[0]
            if polygon.contains(pt):
                points.append((y, x))

    if len(points) == 0:
        c = polygon.centroid
        points = [(c.y, c.x)]

    return points[:n]


rows = []

print(f"TEST RUN: processing {len(tracts)} tracts only")
print()

for i in range(len(tracts)):
    geoid = str(tracts.iloc[i]["GEOID"])
    area = tracts.iloc[i].geometry.area
    n_tiles = tiles_for_area(area)

    polygon_latlon = tracts_latlon.iloc[i].geometry
    points = get_grid_points(polygon_latlon, n_tiles)

    print(
        f"Tract {i} ({geoid}): area={area/1_000_000:.2f} km², requesting {n_tiles} tiles, got {len(points)} valid points"
    )

    for tile_idx, (lat, lon) in enumerate(points):
        try:
            result = get_image(lat, lon, f"{geoid}_{tile_idx}", zoom=ZOOM)
            result["GEOID"] = geoid
            result["tile_idx"] = tile_idx
            result["n_tiles_total"] = len(points)
            rows.append(result)
        except Exception as e:
            print(f"  Failed at tract {i} ({geoid}), tile {tile_idx}: {e}")

image_df = pd.DataFrame(rows)
final_df = image_df.merge(income_df, on="GEOID", how="left")
final_df.to_csv(csv_path, index=False)

print()
print(f"****TEST Dataset saved**** → {csv_path}")
print(f"Total rows: {len(final_df)}")
print()
print("--- Preview ---")
print(final_df.head(10))
print()
print("--- Check for missing incomes (merge issues) ---")
print(final_df["median_income"].isna().sum(), "rows with missing income")
