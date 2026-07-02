import os
import requests

from config import IMAGE_DIR

api_key = os.getenv("GOOGLE_MAPS_API_KEY")

if api_key is None:
    raise ValueError("GOOGLE_MAPS_API_KEY not found.")


def get_image(lat, lon, geoid, zoom=17, size="400x400"):
    
    # Download one satellite image and return metadata.
    

    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}"
        f"&zoom={zoom}"
        f"&size={size}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )

    response = requests.get(url)
    response.raise_for_status()

    filename = IMAGE_DIR / f"{geoid}.png"

    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Saved image → {filename}")

    return {
        "GEOID": geoid,
        "lat": lat,
        "lon": lon,
        "image_path": str(filename)
    }