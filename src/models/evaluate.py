import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def evaluate_model():

    project_root = Path(__file__).resolve().parents[2]

    test_path = (
        project_root
        / "data"
        / "processed"
        / "test.csv"
    )

    model_path = (
        project_root
        / "models"
        / "fraud_model.joblib"
    )

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")

    test_df = pd.read_csv(test_path)

    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    print("Loading trained model...")

    model = joblib.load(model_path)

    print("Generating predictions...")

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4)
    }

    metrics_path = reports_dir / "metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel Evaluation")
    print("=" * 40)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\nMetrics saved to:")
    print(metrics_path)


if __name__ == "__main__":
    evaluate_model()