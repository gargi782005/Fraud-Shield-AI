import pandas as pd
from pathlib import Path


def load_data():
    """Load the raw credit card fraud dataset."""

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "creditcard.csv"

    df = pd.read_csv(data_path)

    return df


if __name__ == "__main__":
    df = load_data()

    print("\nDataset loaded successfully!")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentages:")
    print(df["Class"].value_counts(normalize=True) * 100)