import joblib

from pathlib import Path


# Find project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"


# Load trained model
model = joblib.load(MODEL_PATH)

# Load scaler used during preprocessing
scaler = joblib.load(SCALER_PATH)


def get_model():
    """Return the trained fraud detection model."""
    return model


def get_scaler():
    """Return the preprocessing scaler."""
    return scaler