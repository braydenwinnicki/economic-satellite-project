import pandas as pd  

# read the raw CSV that has all tiles and their (possibly invalid) income labels
df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/raw/ct_tracts_tiles.csv"
)

# strip any whitespace from column names (common issue with CSV exports)
df.columns = df.columns.str.strip()

# -666666666 is a sentinel value meaning "no data" in the Census API
# replace it with pd.NA so dropna() can clean it up properly
df["median_income"] = df["median_income"].replace(-666666666, pd.NA)

# drop rows where income is missing — no point training on tracts without labels
df = df.dropna(subset="median_income")

# save cleaned data to processed directory for use by training/eval scripts
df.to_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts_tiles.csv",
    index=False,
)

print(f"Rows before filtering (already dropped in CSV read): irrelevant, check below")
print(f"Rows after filtering: {len(df)}")
print(f"Unique tracts remaining: {df['GEOID'].nunique()}")

# matplotlib setup if needed
"""
import matplotlib.pyplot as plt

# 1. Create data
x = df.index
y = df.median_income

# 2. Setup canv as (width, height in inches)
fig, ax = plt.subplots(figsize=(7, 4))

# 3. Plot the data
ax.plot(x, y, color="royalblue", linewidth=2)

# 4. Customize labels & titles
ax.set_title("Median Incomes", fontsize=14, fontweight="bold")
ax.set_xlabel("IDX")
ax.set_ylabel("Median Income")
ax.grid(True, linestyle="--", alpha=0.6) # Adds a clean background grid


# 5. Show it
plt.show()
"""