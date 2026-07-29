import os
import time
import requests

from new_pipeline.src.config import IMAGE_DIR


def get_image(lat, lon, geoid, zoom=17, size="400x400", max_retries=5):
    """
    Download one satellite image from the Google Maps Static API and save it.

    Parameters
    ----------
    lat : float
        Latitude of the image center.
    lon : float
        Longitude of the image center.
    geoid : str
        GEOID used to name the saved file (e.g. "09001123400" or "09001123400_0").
    zoom : int, optional
        Google Maps zoom level (default: 17).
    size : str, optional
        Image dimensions as "WxH" (default: "400x400").
    max_retries : int, optional
        Maximum number of retry attempts on HTTP 429/5xx errors
        (default: 5). Uses exponential backoff with jitter.

    Returns
    -------
    dict
        Metadata dict with keys: GEOID, lat, lon, image_path.
    """

    # Check for API key at call time (not import time) so the module
    # remains importable for testing without a key set.
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key is None:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY not found. Set it in your shell profile or .env file."
        )

    filename = IMAGE_DIR / f"{geoid}.png"

    # Skip download if the image already exists on disk (idempotent re-runs)
    if filename.exists():
        print(f"Image already exists, skipping → {filename}")
        return {"GEOID": geoid, "lat": lat, "lon": lon, "image_path": str(filename)}

    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}"
        f"&zoom={zoom}"
        f"&size={size}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )

    # Retry with exponential backoff for rate-limit (429) and server errors (5xx)
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            break  # success — exit the retry loop
        except requests.exceptions.HTTPError as e:
            status_code = (
                e.response.status_code if e.response is not None else "unknown"
            )
            if status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                # Exponential backoff: 2^attempt seconds + small jitter
                wait = 2**attempt + 0.5
                print(
                    f"  HTTP {status_code} for {geoid}, retrying in {wait:.1f}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                raise
    else:
        # This branch runs if the loop exhausted all retries without breaking
        raise RuntimeError(
            f"Failed to download image for {geoid} after {max_retries} attempts."
        )

    # Open the file in write-binary mode ("wb") and save the image bytes
    # response.content contains the raw bytes of the downloaded image (PNG)
    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Saved image → {filename}")

    # Return a dict with metadata that will become one row in the CSV
    return {"GEOID": geoid, "lat": lat, "lon": lon, "image_path": str(filename)}
