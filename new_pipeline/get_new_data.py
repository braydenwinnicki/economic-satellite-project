"""
get_new_data.py — User-facing entry point for the data pipeline.

Usage
-----
    # Single-tile (one centroid image per census tract):
    python3 new_pipeline/get_new_data.py \
        --shapefile /path/to/tracts.shp \
        --fips 09 \
        --census-api-link "https://api.census.gov/data/2023/acs/acs5"

    # Multi-tile (multiple grid-sampled images per tract):
    python3 new_pipeline/get_new_data.py \
        --shapefile "/Users/braydenwinnicki/Downloads/cb_2025_09_tract_500k/cb_2025_09_tract_500k.shp" \
        --fips 09 \
        --mode multi \
        --census-api-link "https://api.census.gov/data/2023/acs/acs5"

The pipeline:
  1. Reads the census tract shapefile
  2. Downloads satellite imagery for each tract centroid (single-tile)
     or grid-sampled points across each tract (multi-tile)
  3. Fetches median household income for each tract (Census API)
  4. Merges everything into a raw CSV
  5. Preprocesses the CSV (removes sentinel values, drops missing rows)
  6. Builds a PyTorch cache (.pt) in the appropriate format
"""

import argparse
import sys
from pathlib import Path

# Path(__file__) = this script's path; .resolve() = absolute path; .parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from new_pipeline.build_csv import build_csv, build_csv_multi
from new_pipeline.src.build_cache import build_cache, build_tract_cache
from new_pipeline.src.config import CACHE_DIR, DATA_DIR
from new_pipeline.src.preprocessing import preprocess_data


def main():
    parser = argparse.ArgumentParser(
        description="Download satellite imagery + census data for a given state."
    )
    parser.add_argument(
        "--shapefile",
        required=True,
        help="Path to the census tract shapefile (.shp)",
    )
    parser.add_argument(
        "--fips",
        required=True,
        help="State FIPS code (e.g. '09' for Connecticut)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Data collection mode: 'single' = one centroid image per tract, "
        "'multi' = multiple grid-sampled images per tract (default: single)",
    )
    parser.add_argument(
        "--census-api-link",
        default="https://api.census.gov/data/2023/acs/acs5",
        help="Census API base URL (default: 2023 ACS 5-year estimates)",
    )

    args = parser.parse_args()

    is_multi = args.mode == "multi"
    suffix = "_multi" if is_multi else ""

    # File paths — multi-tile uses _multi suffix to avoid collisions
    raw_csv = DATA_DIR / f"{args.fips}_tracts{suffix}.csv"
    processed_csv = DATA_DIR / f"processed_{args.fips}_tracts{suffix}.csv"
    cache_file = CACHE_DIR / f"{args.fips}_tracts{suffix}.pt"

    # Step 1: Build raw CSV + download images

    print("=" * 60)
    print(
        f"Step 1: Building raw dataset ({'multi-tile' if is_multi else 'single-tile'} mode)..."
    )
    print("=" * 60)
    if is_multi:
        build_csv_multi(args.shapefile, args.fips, args.census_api_link)
    else:
        build_csv(args.shapefile, args.fips, args.census_api_link)

    # Step 2: Preprocess the CSV

    print("\n" + "=" * 60)
    print("Step 2: Preprocessing CSV...")
    print("=" * 60)

    preprocess_data(raw_csv, processed_csv)

    # Step 3: Build PyTorch cache

    print("\n" + "=" * 60)
    print(
        f"Step 3: Building PyTorch cache ({'multi-tile' if is_multi else 'single-tile'} format)..."
    )
    print("=" * 60)

    if is_multi:
        build_tract_cache(processed_csv, cache_file)
    else:
        build_cache(processed_csv, cache_file)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Raw CSV:       {raw_csv}")
    print(f"  Processed CSV: {processed_csv}")
    print(f"  Cache:         {cache_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
