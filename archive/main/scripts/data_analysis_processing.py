import pandas as pd

df = pd.read_csv("/Users/braydenwinnicki/CODE/econ_project/data/raw/ct_tracts.csv")

# .columns returns all column names as an Index
# .str.strip() removes leading/trailing whitespace from each column name
df.columns = df.columns.str.strip()


# -666666666 is a sentinel value used by the Census Bureau to mean "no data"
# .replace(-666666666, pd.NA) replaces that sentinel with a proper missing value
# pd.NA is pandas' own missing value type (type-aware, preserves dtypes)
df["median_income"] = df["median_income"].replace(-666666666, pd.NA)

# .dropna(subset="median_income") removes any row where median_income is NA
# subset="median_income" limits the NA check to just that one column
df = df.dropna(subset="median_income")


# .to_csv() writes the DataFrame to a CSV file
# index=False means don't write row numbers as a separate column
df.to_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed_ct_tracts.csv", index=False
)
