from pathlib import Path
import joblib


def test_model_exists():
    model_path = Path("models/fraud_model.joblib")

    assert model_path.exists()


def test_model_can_be_loaded():
    model_path = Path("models/fraud_model.joblib")

    model = joblib.load(model_path)

    assert model is not None