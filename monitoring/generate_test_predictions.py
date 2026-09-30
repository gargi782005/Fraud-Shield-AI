import pandas as pd
import requests
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

API_URL = "http://127.0.0.1:8000/predict"

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


def main():

    if not TEST_DATA.exists():
        print("Test dataset not found.")
        return

    df = pd.read_csv(TEST_DATA)

    # Select 30 transactions
    sample = df.sample(
    n=500,
    random_state=42
)

    print(f"Sending {len(sample)} transactions to FastAPI...\n")

    successful = 0

    for index, row in sample.iterrows():

        transaction = {
            column: float(row[column])
            for column in FEATURE_COLUMNS
        }

        try:
            response = requests.post(
                API_URL,
                json=transaction,
                timeout=10
            )

            response.raise_for_status()

            result = response.json()

            successful += 1

            print(
                f"Transaction {index + 1}: "
                f"{result['result']} | "
                f"Fraud probability: "
                f"{result['fraud_probability']}"
            )

        except requests.exceptions.RequestException as error:
            print(
                f"Transaction {index + 1} failed: {error}"
            )

    print("\n" + "=" * 50)
    print(f"Successful predictions: {successful}")
    print("=" * 50)


if __name__ == "__main__":
    main()