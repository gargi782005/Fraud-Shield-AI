import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


RANDOM_STATE = 42


def load_data():
    """Load the raw credit card fraud dataset."""

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "creditcard.csv"

    return pd.read_csv(data_path)


def preprocess_data(df):
    """
    Prepare the dataset for machine learning.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split the data while preserving the fraud/legitimate ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scale Time and Amount.
    # V1-V28 are already transformed numerical features.
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )

    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    return X_train, X_test, y_train, y_test, scaler


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler
):
    """Save processed datasets and scaler."""

    project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    train_data["Class"] = y_train.values

    test_data = X_test.copy()
    test_data["Class"] = y_test.values

    train_data.to_csv(
        processed_dir / "train.csv",
        index=False
    )

    test_data.to_csv(
        processed_dir / "test.csv",
        index=False
    )

    joblib.dump(
        scaler,
        models_dir / "scaler.joblib"
    )

    print("\nProcessed data saved successfully!")
    print(f"Training data: {train_data.shape}")
    print(f"Testing data: {test_data.shape}")
    print(f"Scaler: {models_dir / 'scaler.joblib'}")


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data()

    print(f"Original dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("\nPreprocessing completed.")

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )