import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42


def load_training_data():

    project_root = Path(__file__).resolve().parents[2]

    train_path = (
        project_root
        / "data"
        / "processed"
        / "train.csv"
    )

    train_df = pd.read_csv(train_path)

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]

    return X_train, y_train


def train_and_save_model():

    project_root = Path(__file__).resolve().parents[2]

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")

    X_train, y_train = load_training_data()

    print(f"Training samples: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    model_path = models_dir / "fraud_model.joblib"

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved successfully!")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    train_and_save_model()