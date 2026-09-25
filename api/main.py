import pandas as pd

from fastapi import FastAPI

from api.model_loader import get_model, get_scaler
from api.schemas import (
    TransactionRequest,
    PredictionResponse
)


app = FastAPI(
    title="Fraud Shield AI API",
    description="AI-powered online transaction fraud detection API",
    version="1.0.0"
)


model = get_model()
scaler = get_scaler()


@app.get("/")
def root():
    return {
        "message": "Fraud Shield AI API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Random Forest"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(transaction: TransactionRequest):

    # Convert request to dictionary
    data = transaction.model_dump()

    # Create DataFrame
    df = pd.DataFrame([data])

    # Scale Time and Amount
    df[["Time", "Amount"]] = scaler.transform(
        df[["Time", "Amount"]]
    )

    # Make prediction
    prediction = int(model.predict(df)[0])

    # Fraud probability
    probability = float(
        model.predict_proba(df)[0][1]
    )

    if prediction == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Legitimate Transaction"

    return PredictionResponse(
        prediction=prediction,
        result=result,
        fraud_probability=round(probability, 4)
    )