import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_tract(df: pd.DataFrame, test_size: float = 0.2, random_state=42):
    """Split a tile-level dataframe by tract so all tiles from one tract stay together."""
    if "GEOID" not in df.columns:
        raise KeyError("Input dataframe must include a GEOID column.")

    tract_ids = df["GEOID"].drop_duplicates().to_numpy()
    train_tracts, test_tracts = train_test_split(
        tract_ids,
        test_size=test_size,
        random_state=random_state,
    )

    train_df = df[df["GEOID"].isin(train_tracts)].copy()
    test_df = df[df["GEOID"].isin(test_tracts)].copy()

    return train_df, test_df
