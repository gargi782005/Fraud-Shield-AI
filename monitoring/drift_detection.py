from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DATA = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PREDICTION_LOG = PROJECT_ROOT / "monitoring" / "prediction_logs.csv"

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

DRIFT_THRESHOLD = 0.25
MIN_NEW_RECORDS = 30


def calculate_psi(reference, current, bins=10):

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    breakpoints = np.unique(
        np.percentile(
            reference,
            np.linspace(0, 100, bins + 1)
        )
    )

    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts = pd.cut(
        reference,
        bins=breakpoints,
        include_lowest=True
    ).value_counts(sort=False)

    current_counts = pd.cut(
        current,
        bins=breakpoints,
        include_lowest=True
    ).value_counts(sort=False)

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 0.0001

    reference_pct = reference_pct.clip(lower=epsilon)
    current_pct = current_pct.clip(lower=epsilon)

    psi = (
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    ).sum()

    return float(psi)


def detect_drift():

    print("\n" + "=" * 60)
    print("           FRAUD SHIELD AI - DATA DRIFT")
    print("=" * 60)

    # Check reference dataset
    if not REFERENCE_DATA.exists():

        print("Reference dataset not found.")
        print(f"Expected location: {REFERENCE_DATA}")

        return None

    # Check prediction logs
    if not PREDICTION_LOG.exists():

        print("Prediction log not found.")
        print(f"Expected location: {PREDICTION_LOG}")
        print("Prediction logs are required for drift detection.")

        return None

    # Load datasets
    reference_df = pd.read_csv(REFERENCE_DATA)
    current_df = pd.read_csv(PREDICTION_LOG)

    print(f"Reference records : {len(reference_df):,}")
    print(f"New API records   : {len(current_df):,}")

    # Minimum records check
    if len(current_df) < MIN_NEW_RECORDS:

        print("\nNot enough new records for reliable drift detection.")

        print(f"Minimum required : {MIN_NEW_RECORDS}")
        print(f"Currently have   : {len(current_df)}")

        print("\nMake more predictions through the API.")

        return None

    drift_results = []

    # Calculate PSI for every feature
    for feature in FEATURE_COLUMNS:

        if feature not in reference_df.columns:
            continue

        if feature not in current_df.columns:
            continue

        psi_value = calculate_psi(
            reference_df[feature],
            current_df[feature]
        )

        if psi_value >= DRIFT_THRESHOLD:
            status = "DRIFT"
        else:
            status = "STABLE"

        drift_results.append({
            "feature": feature,
            "psi": round(psi_value, 4),
            "status": status
        })

    results_df = pd.DataFrame(drift_results)

    drifted_features = results_df[
        results_df["status"] == "DRIFT"
    ]

    print("\nFeature Drift Results")
    print("-" * 60)

    print(
        results_df.to_string(index=False)
    )

    print("\n" + "-" * 60)

    # Drift detected
    if len(drifted_features) > 0:

        print("⚠️ DATA DRIFT DETECTED")

        print(
            f"Drifted features: "
            f"{len(drifted_features)} / {len(results_df)}"
        )

        print(
            "\nRetraining should be considered."
        )

        return True

    # No drift
    else:

        print("✅ NO SIGNIFICANT DATA DRIFT")

        print(
            "Current transaction data is within the monitoring threshold."
        )

        return False


if __name__ == "__main__":

    drift_detected = detect_drift()

    if drift_detected is True:

        # Exit code 1 = drift detected
        raise SystemExit(1)

    elif drift_detected is False:

        # Exit code 0 = no drift
        raise SystemExit(0)

    else:

        # Exit code 2 = monitoring data/configuration problem
        raise SystemExit(2)