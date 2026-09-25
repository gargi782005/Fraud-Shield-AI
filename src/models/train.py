import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


RANDOM_STATE = 42


def load_processed_data():
    """Load the processed training and testing datasets."""

    project_root = Path(__file__).resolve().parents[2]

    train_path = project_root / "data" / "processed" / "train.csv"
    test_path = project_root / "data" / "processed" / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]

    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    return X_train, X_test, y_train, y_test


def train_hist_gradient_boosting():

    X_train, X_test, y_train, y_test = load_processed_data()

    # Use the same MLflow experiment
    mlflow.set_experiment("Fraud Shield AI")

    with mlflow.start_run(run_name="HistGradientBoosting"):

        # Create model
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE
        )

        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluation metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        # Log parameters
        mlflow.log_param(
            "model",
            "HistGradientBoosting"
        )

        mlflow.log_param(
            "max_iter",
            200
        )

        mlflow.log_param(
            "learning_rate",
            0.1
        )

        mlflow.log_param(
            "max_leaf_nodes",
            31
        )

        # Log metrics
        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.log_metric(
            "pr_auc",
            pr_auc
        )

        # Log model
        mlflow.sklearn.log_model(
    model,
    name="hist_gradient_boosting_model",
    skops_trusted_types=[
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
    ]
)

        print("\nHistGradientBoosting Results")
        print("-" * 40)

        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"PR-AUC    : {pr_auc:.4f}")

        print("\nMLflow run completed successfully.")


if __name__ == "__main__":
    train_hist_gradient_boosting()