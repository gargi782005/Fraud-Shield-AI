import pandas as pd
from pathlib import Path


def create_model_comparison():

    project_root = Path(__file__).resolve().parents[2]

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = [
        {
            "Model": "Logistic Regression",
            "Precision": 0.0610,
            "Recall": 0.9184,
            "F1 Score": 0.1144,
            "ROC-AUC": 0.9722,
            "PR-AUC": 0.7159
        },
        {
            "Model": "Random Forest",
            "Precision": 0.7168,
            "Recall": 0.8265,
            "F1 Score": 0.7678,
            "ROC-AUC": 0.9790,
            "PR-AUC": 0.8236
        },
        {
            "Model": "HistGradientBoosting",
            "Precision": 0.5455,
            "Recall": 0.7347,
            "F1 Score": 0.6261,
            "ROC-AUC": 0.8075,
            "PR-AUC": 0.5679
        }
    ]

    df = pd.DataFrame(results)

    output_path = reports_dir / "model_comparison.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print("\nModel Comparison")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\nComparison saved successfully!")
    print(f"File: {output_path}")


if __name__ == "__main__":
    create_model_comparison()