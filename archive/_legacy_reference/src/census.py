import os
import requests
import pandas as pd

# load Census API key from environment
api_key = os.getenv("CENSUS_API_KEY")

if api_key is None:
    raise ValueError("CENSUS_API_KEY not found.")


def get_income_date():

    # request median household income (B19013_001E) for all tracts in state 09
    params = {"get": "B19013_001E", "for": "tract:*", "in": "state:09", "key": api_key}
    response = requests.get("https://api.census.gov/data/2023/acs/acs5", params=params)

    # raise an exception if the request failed
    response.raise_for_status()

    data = response.json()

    # first row contains column names, remaining rows contain values
    columns = data[0]
    rows = data[1:]

    # build a DataFrame and create a GEOID by concatenating state/county/tract
    df = pd.DataFrame(rows, columns=columns)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    # rename Census variable to an intuitive column name
    df = df.rename(columns={"B19013_001E": "median_income"})

    # keep only the GEOID and median_income columns
    df = df[["GEOID", "median_income"]]

    return df
