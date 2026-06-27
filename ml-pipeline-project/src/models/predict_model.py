"""
predict_model.py
Loads the trained model and runs predictions on new data.
"""
import pandas as pd
import joblib
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str):
    return joblib.load(model_path)


def predict(model, input_df: pd.DataFrame):
    return model.predict(input_df)


if __name__ == "__main__":
    config = load_config()
    model = load_model(config["model"]["save_path"])

    sample = pd.read_csv("data/processed/X_test.csv").iloc[:5]
    preds = predict(model, sample)
    print("Sample predictions:", preds)
