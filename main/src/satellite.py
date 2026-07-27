import os
import requests

from src.config import IMAGE_DIR

# os.getenv() reads an environment variable from your system
# You need to set GOOGLE_MAPS_API_KEY in your shell profile or .env file
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

if api_key is None:
    raise ValueError("GOOGLE_MAPS_API_KEY not found.")


def get_image(lat, lon, geoid, zoom=17, size="400x400"):

    # Download one satellite image and return metadata.

    # f-strings let you embed variables directly into strings with {variable}
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}"
        f"&zoom={zoom}"
        f"&size={size}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )

    # requests.get() makes an HTTP GET request to the URL
    response = requests.get(url)

    # .raise_for_status() throws an HTTPError if the status is 400 or higher
    response.raise_for_status()

    # IMAGE_DIR / f"{geoid}.png" constructs a file path using pathlib
    # f"{geoid}.png" creates a filename like "09001123400.png"
    filename = IMAGE_DIR / f"{geoid}.png"

    # Open the file in write-binary mode ("wb") and save the image bytes
    # response.content contains the raw bytes of the downloaded image (PNG)
    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Saved image → {filename}")

    # Return a dict with metadata that will become one row in the CSV
    return {"GEOID": geoid, "lat": lat, "lon": lon, "image_path": str(filename)}