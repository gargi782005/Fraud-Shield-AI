from unittest.mock import patch

from fastapi.testclient import TestClient


class DummyModel:

    def predict(self, data):
        return [0]

    def predict_proba(self, data):
        return [[0.99, 0.01]]


class DummyScaler:

    def transform(self, data):
        return data


with patch(
    "joblib.load",
    side_effect=[
        DummyModel(),
        DummyScaler()
    ]
):
    from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Fraud Shield AI API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"