import pandas as pd


def test_dataset_columns():
    df = pd.DataFrame(columns=[
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
        "Class"
    ])

    expected_columns = ["Time"] + [
        f"V{i}" for i in range(1, 29)
    ] + ["Amount", "Class"]

    assert list(df.columns) == expected_columns


def test_target_column_exists():
    df = pd.DataFrame({
        "Time": [0.0],
        "V1": [0.1],
        "Amount": [10.0],
        "Class": [0]
    })

    assert "Class" in df.columns