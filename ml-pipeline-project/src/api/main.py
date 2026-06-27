"""
main.py
FastAPI application that serves the trained ML model as a REST API.
"""
import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from src.api.schemas import PredictionInput, PredictionOutput, BatchPredictionInput

app = FastAPI(
    title="ML Pipeline API",
    description="Serves predictions from a trained scikit-learn model",
    version="1.0.0"
)

# Load config and model once at startup
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = joblib.load(config["model"]["save_path"])


@app.get("/")
def root():
    return {"status": "API is running", "model": config["model"]["type"]}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        prediction = model.predict(df)[0]
        probability = max(model.predict_proba(df)[0])
        return PredictionOutput(prediction=int(prediction), probability=float(probability))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(input_data: BatchPredictionInput):
    try:
        df = pd.DataFrame([r.dict() for r in input_data.records])
        predictions = model.predict(df)
        probabilities = model.predict_proba(df).max(axis=1)
        return [
            {"prediction": int(p), "probability": float(prob)}
            for p, prob in zip(predictions, probabilities)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
