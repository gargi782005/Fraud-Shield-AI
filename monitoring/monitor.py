import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prediction log file
LOG_FILE = PROJECT_ROOT / "monitoring" / "prediction_logs.csv"


def monitor_predictions():

    # Check whether prediction log exists
    if not LOG_FILE.exists():
        print("No prediction log found.")
        print("Make at least one prediction through the API first.")
        return

    # Read prediction logs
    df = pd.read_csv(LOG_FILE)

    # Total predictions
    total_predictions = len(df)

    # Fraud predictions
    fraud_predictions = (df["prediction"] == 1).sum()

    # Legitimate predictions
    legitimate_predictions = (df["prediction"] == 0).sum()

    # Average latency
    average_latency = df["latency_ms"].mean()

    # Maximum latency
    maximum_latency = df["latency_ms"].max()

    # Display monitoring results
    print("\n" + "=" * 50)
    print("       FRAUD SHIELD AI - MONITORING")
    print("=" * 50)

    print(f"Total Predictions       : {total_predictions}")
    print(f"Fraud Predictions       : {fraud_predictions}")
    print(f"Legitimate Predictions  : {legitimate_predictions}")
    print(f"Average Latency         : {average_latency:.2f} ms")
    print(f"Maximum Latency         : {maximum_latency:.2f} ms")

    print("=" * 50)


if __name__ == "__main__":
    monitor_predictions()