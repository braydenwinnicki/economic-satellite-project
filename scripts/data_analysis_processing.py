import pandas as pd

df = pd.read_csv("/Users/braydenwinnicki/CODE/econ_project/data/raw/ct_tracts.csv")

df.columns = df.columns.str.strip()



df["median_income"] = df["median_income"].replace(-666666666, pd.NA)

df = df.dropna(subset="median_income")




df.to_csv("/Users/braydenwinnicki/CODE/econ_project/data/processed_ct_tracts.csv", index=False)







#matplotlib setup if needed
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