import pandas as pd


def preprocess_data(input_csv, output_csv):
    """
    Clean a raw tracts CSV by:
      1. Stripping whitespace from column names
      2. Replacing Census sentinel value -666666666 with pd.NA
      3. Dropping rows with missing median_income

    Parameters
    ----------
    input_csv : str or Path
        Path to the raw CSV produced by build_csv().
    output_csv : str or Path
        Path where the cleaned CSV will be saved.
    """
    df = pd.read_csv(input_csv)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # -666666666 is a sentinel value used by the Census Bureau to mean "no data"
    # Replace it with a proper missing value
    df["median_income"] = df["median_income"].replace(-666666666, pd.NA)

    # Drop any row where median_income is NA
    df = df.dropna(subset="median_income")

    # Save the cleaned dataset
    df.to_csv(output_csv, index=False)

    print(f"Preprocessed data saved → {output_csv}")
    print(f"  Rows before: {len(pd.read_csv(input_csv))}")
    print(f"  Rows after:  {len(df)}")

    return output_csv
