import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts.csv"
)

# calculate MAE uing averages
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

mean_income = df_train["median_income"].mean()

baseline_predictions = [mean_income] * len(df_test)

baseline_mae = mean_absolute_error(df_test["median_income"], baseline_predictions)

print(baseline_mae)
