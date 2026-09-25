import pandas as pd


def test_dataset_columns():
    df = pd.read_csv("data/raw/creditcard.csv")

    expected_columns = ["Time"] + [
        f"V{i}" for i in range(1, 29)
    ] + ["Amount", "Class"]

    assert list(df.columns) == expected_columns


def test_target_column_exists():
    df = pd.read_csv("data/raw/creditcard.csv")

    assert "Class" in df.columns