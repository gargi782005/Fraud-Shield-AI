import pandas as pd
from pathlib import Path


def select_best_model():

    project_root = Path(__file__).resolve().parents[2]

    comparison_path = (
        project_root
        / "reports"
        / "model_comparison.csv"
    )

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load model comparison results
    df = pd.read_csv(comparison_path)

    # Select model using F1 Score
    best_row = df.loc[df["F1 Score"].idxmax()]

    best_model = best_row["Model"]
    best_f1 = best_row["F1 Score"]

    # Save selection information
    selection = pd.DataFrame([
        {
            "Selected Model": best_model,
            "Selection Metric": "F1 Score",
            "F1 Score": best_f1
        }
    ])

    output_path = reports_dir / "best_model.csv"

    selection.to_csv(
        output_path,
        index=False
    )

    print("\nBest Model Selection")
    print("=" * 50)

    print(f"Selected Model : {best_model}")
    print(f"Selection Metric: F1 Score")
    print(f"F1 Score       : {best_f1:.4f}")

    print("\nSelection saved successfully!")
    print(f"File: {output_path}")


if __name__ == "__main__":
    select_best_model()