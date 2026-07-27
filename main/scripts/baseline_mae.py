import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts.csv"
)

# calculate MAE uing averages

# train_test_split splits the DataFrame into two random subsets
# test_size=0.2 means 20% of rows go to test, 80% to train
# random_state=42 makes the split reproducible (same split every time)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# .mean() computes the average of the "median_income" column across all training rows
mean_income = df_train["median_income"].mean()

# Create a list of predictions where every entry is the mean income
# [mean_income] * len(df_test) repeats the same value N times (one per test row)
baseline_predictions = [mean_income] * len(df_test)

# mean_absolute_error computes the average absolute difference between actual and predicted
# This gives us a baseline: if we just guessed the mean every time, how wrong would we be?
baseline_mae = mean_absolute_error(df_test["median_income"], baseline_predictions)

print(baseline_mae)