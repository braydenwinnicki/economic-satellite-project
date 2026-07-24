import os
import requests
import pandas as pd

api_key = os.getenv("CENSUS_API_KEY")

if api_key is None:
    raise ValueError("CENSUS_API_KEY not found.")


def get_income_date():

    # access census api
    params = {"get": "B19013_001E", "for": "tract:*", "in": "state:09", "key": api_key}
    response = requests.get("https://api.census.gov/data/2023/acs/acs5", params=params)

    response.raise_for_status()

    data = response.json()

    columns = data[0]

    rows = data[1:]

    df = pd.DataFrame(rows, columns=columns)

    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    df = df.rename(columns={"B19013_001E": "median_income"})

    df = df[["GEOID", "median_income"]]

    return df
