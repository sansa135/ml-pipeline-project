"""
test_api.py
Basic unit tests for the FastAPI app.
Run with: pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_input():
    payload = {
        "feature_1": 0.5,
        "feature_2": 1.2,
        "feature_3": 3.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_invalid_input():
    payload = {"feature_1": "invalid"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
