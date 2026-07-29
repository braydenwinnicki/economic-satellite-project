import os
import requests
import pandas as pd


def get_income_data(fips_code, census_api_link):

    # os.getenv() reads an environment variable from your system
    # You need to set CENSUS_API_KEY in your shell profile or .env file
    api_key = os.getenv("CENSUS_API_KEY")

    if api_key is None:
        raise ValueError("CENSUS_API_KEY not found.")

    # access census api
    # params dict gets converted to URL query parameters by requests.get()
    # "get": "B19013_001E" — Census variable code for median household income
    # "for": "tract:*" — request data for all census tracts
    # "in": "state:XX" — filter to the specified FIPS code
    params = {
        "get": "B19013_001E",
        "for": "tract:*",
        "in": "state:" + fips_code,
        "key": api_key,
    }
    response = requests.get(census_api_link, params=params)

    # .raise_for_status() throws an HTTPError if the status code is 400 or higher
    # This catches bad requests, API errors, etc.
    response.raise_for_status()

    # .json() parses the response body from JSON into a Python list of lists
    # The Census API returns: first row = column names, rest = data rows
    data = response.json()

    # data[0] is the header row: ["B19013_001E", "state", "county", "tract"]
    columns = data[0]

    # data[1:] is everything after the header — the actual data rows
    rows = data[1:]

    # pd.DataFrame(rows, columns=columns) creates a table from the data
    df = pd.DataFrame(rows, columns=columns)

    # Concatenate state + county + tract to form a unique GEOID
    # Example: "09" + "001" + "123400" = "09001123400"
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    # Rename the census variable code to something readable
    df = df.rename(columns={"B19013_001E": "median_income"})

    # Keep only the columns we need
    df = df[["GEOID", "median_income"]]

    return df
