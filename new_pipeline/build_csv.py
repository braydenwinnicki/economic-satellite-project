"""
build_csv.py — Download satellite images and build tile-level CSVs.

Two modes:
  - Single-tile: one centroid image per census tract (build_csv)
  - Multi-tile: multiple grid-sampled images per tract (build_csv_multi)

Both modes:
  1. Read a census-tract shapefile
  2. Compute sampling points (centroid or grid)
  3. Download satellite images via the Google Maps Static API
  4. Merge with Census income data
  5. Save as a CSV with one row per tile

Requires the GOOGLE_MAPS_API_KEY environment variable to be set.
"""

import sys
from pathlib import Path
import pandas as pd

# Path(__file__) = this script's path; .resolve() = absolute path; .parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
import numpy as np

from new_pipeline.src.satellite import get_image
from new_pipeline.src.config import PROJECT_ROOT
from new_pipeline.src.get_incomes import get_income_data


MIN_TILES = 4
MAX_TILES = 50


def tiles_for_area(area_sq_m):
    """Scale tile count with the square root of area (so tile count
    scales with linear dimension, not raw area - keeps things from
    exploding for huge rural tracts).

    Parameters
    ----------
    area_sq_m : float
        Tract area in square meters (after reprojection to EPSG:3857).

    Returns
    -------
    int
        Number of tiles to sample, clamped between MIN_TILES and MAX_TILES.
    """
    area_sq_km = area_sq_m / 1_000_000  # convert to km for ease
    n = (
        int(np.sqrt(area_sq_km)) * 3 + MIN_TILES
    )  # use sqrt to focus on linear side lengths so crazy areas dont mess things up, multiply by 3 to get more coverage
    return max(
        MIN_TILES, min(MAX_TILES, n)
    )  # if n is in range return it, if not send back other option


def get_grid_points(polygon, n):
    """Return up to n points spread across the polygon's bounding box,
    keeping only ones that fall inside the actual tract shape.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The tract polygon (in lat/lon coordinates).
    n : int
        Desired number of sample points.

    Returns
    -------
    list of (float, float)
        List of (lat, lon) tuples that fall within the polygon.
    """
    minx, miny, maxx, maxy = (
        polygon.bounds
    )  # get x,y of smallest rectangle that can fully fit the tract
    side = int(
        np.ceil(np.sqrt(n))
    )  # we need to make n points arranged a sgird so use square root to grab that, round up

    xs = np.linspace(minx, maxx, side + 2)[
        1:-1
    ]  # generates side # of a evenly spaced points on x axis. cuts off forst and alst becuase they are unlikley to be in the actual tract
    ys = np.linspace(miny, maxy, side + 2)[1:-1]  # does same for y

    points = []

    for x in xs:  # loop over each point on x axis
        for y in ys:  # check each y value, creating a grid
            pt = gpd.points_from_xy([x], [y])[
                0
            ]  # convert to gpd spatial point for checks
            if polygon.contains(
                pt
            ):  # confirms the spacial point is inside the actually tract
                points.append((y, x))  # save the point

    if len(points) == 0:  # if its zero, give us at least one photo
        c = polygon.centroid
        points = [(c.y, c.x)]

    return points[:n]  # trim down in case grid made it go over


def build_csv_multi(shapefile_path, fips_code, census_api_link):
    """Build a CSV with multiple tiles per tract using grid sampling.

    Uses a grid-based sampling approach to select image coordinates within
    each tract polygon. Tile count scales with tract area (via sqrt) to
    provide proportionate coverage. Saves checkpoints every 500 rows to
    avoid data loss on long downloads.

    Parameters
    ----------
    shapefile_path : str or Path
        Path to the census tract shapefile (.shp).
    fips_code : str
        State FIPS code (e.g. "09" for Connecticut).
    census_api_link : str
        Census API base URL for income data.

    Notes
    -----
    The CSV is saved to ``{PROJECT_ROOT}/data/{fips_code}_tracts_multi.csv``
    and includes columns: GEOID, tile_idx, n_tiles_total, lat, lon,
    image_path, median_income.
    """
    # get median_incomes from census api
    income_df = get_income_data(fips_code, census_api_link)

    # get tracts for google maps api
    tracts = gpd.read_file(shapefile_path)

    csv_path = PROJECT_ROOT / "data" / f"{fips_code}_tracts_multi.csv"

    # reproject to a meters-based CRS so area is in real units, not degrees
    tracts = tracts.to_crs(epsg=3857)

    # keep original (lat/lon) geometry around for get_image, since we reprojected above
    tracts_latlon = gpd.read_file(shapefile_path)

    # build dataset
    rows = []

    for i in range(len(tracts)):  # grab each tract
        geoid = str(tracts.iloc[i]["GEOID"])  # get its GEOID
        area = tracts.iloc[i].geometry.area  # in sq meters now, thanks to reprojection
        n_tiles = tiles_for_area(area)  # get amount of pics needed for tract

        # sample points using the lat/lon version (Google Maps wants lat/lon, not meters)
        polygon_latlon = tracts_latlon.iloc[i].geometry  # grab this tracts lat/lon
        points = get_grid_points(
            polygon_latlon, n_tiles
        )  # create the grid points for pic locations

        for tile_idx, (lat, lon) in enumerate(points):  # for each tile(with locations)
            try:
                result = get_image(lat, lon, f"{geoid}_{tile_idx}")  # get_image
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


def build_csv(shapefile_path, fips_code, census_api_link):
    """Build a CSV with one image per tract (centroid).

    For each tract in the shapefile, computes the centroid point and
    downloads a single satellite image. Saves checkpoints every 100 rows.

    Parameters
    ----------
    shapefile_path : str or Path
        Path to the census tract shapefile (.shp).
    fips_code : str
        State FIPS code (e.g. "09" for Connecticut).
    census_api_link : str
        Census API base URL for income data.

    Notes
    -----
    The CSV is saved to ``{PROJECT_ROOT}/data/{fips_code}_tracts.csv``
    and includes columns: GEOID, lat, lon, image_path, median_income.
    """
    # get median_incomes from census api
    income_df = get_income_data(fips_code, census_api_link)

    # get tracts for google maps api
    # gpd.read_file loads a shapefile (a geospatial data format) into a GeoDataFrame
    # Each row in the shapefile is one census tract with its boundary polygon
    tracts = gpd.read_file(shapefile_path)

    csv_path = PROJECT_ROOT / "data" / f"{fips_code}_tracts.csv"

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