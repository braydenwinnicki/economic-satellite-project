import os  # for environment variables
import requests  # HTTP requests to Google Static Maps API

from src.config import IMAGE_DIR  # directory to save downloaded images

# load Google Maps API key from environment
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

if api_key is None:
    # fail early if API key not configured
    raise ValueError("GOOGLE_MAPS_API_KEY not found.")


def get_image(lat, lon, geoid, zoom=17, size="400x400"):

    # construct the Google Static Maps URL for a satellite image
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}"
        f"&zoom={zoom}"
        f"&size={size}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )

    # download the image and raise if non-2xx
    response = requests.get(url)
    response.raise_for_status()

    # save to disk under IMAGE_DIR with GEOID-based filename
    filename = IMAGE_DIR / f"{geoid}.png"

    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Saved image → {filename}")

    # return metadata row suitable for merging into dataset CSV
    return {"GEOID": geoid, "lat": lat, "lon": lon, "image_path": str(filename)}
