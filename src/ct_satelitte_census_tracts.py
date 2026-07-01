import geopandas as gpd
import requests
from pathlib import Path

# Paths (stable project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Load shapefile 
tracts = gpd.read_file(
    "/Users/braydenwinnicki/Downloads/cb_2025_09_bg_500k/cb_2025_09_bg_500k.shp"
)


def get_image(row_id, zoom: int = 18, size: str = "400x400"):
    """
    Fetch satellite image for a single census tract row index.
    Saves image to data/raw/images using GEOID as filename.
    """

    current_tract = tracts.iloc[row_id]

    # centroid
    centroid = current_tract.geometry.centroid
    lat = centroid.y
    lon = centroid.x

    # API key (IMPORTANT: later move to environment variable)
    api_key = "AIzaSyCrZywR0a_2h2KuAHki7HdxOopej1aDTOA"

    # build request URL
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}"
        f"&zoom={zoom}"
        f"&size={size}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )

    # request image
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")

    # safe filename 
    geoid = str(current_tract["GEOID"])
    filename = IMAGE_DIR / f"{geoid}.png"

    # save image
    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Saved image → {filename}")

    return filename


get_image(1)