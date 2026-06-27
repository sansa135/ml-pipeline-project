"""
schemas.py
Pydantic models for API request and response validation.
Adjust fields to match your actual dataset's feature columns.
"""
from pydantic import BaseModel
from typing import List


class PredictionInput(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    # Add the rest of your model's input features here
    # Example:
    # tenure_months: float
    # monthly_charges: float
    # contract_type: int

    class Config:
        json_schema_extra = {
            "example": {
                "feature_1": 0.5,
                "feature_2": 1.2,
                "feature_3": 3.4
            }
        }


class PredictionOutput(BaseModel):
    prediction: int
    probability: float


class BatchPredictionInput(BaseModel):
    records: List[PredictionInput]
