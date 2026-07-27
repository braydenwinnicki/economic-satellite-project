import pandas as pd

df = pd.read_csv("/Users/braydenwinnicki/CODE/econ_project/data/raw/ct_tracts.csv")

df.columns = df.columns.str.strip()


df["median_income"] = df["median_income"].replace(-666666666, pd.NA)

df = df.dropna(subset="median_income")


df.to_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed_ct_tracts.csv", index=False
)


