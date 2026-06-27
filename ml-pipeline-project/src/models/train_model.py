"""
train_model.py
Trains a model on processed data and saves it to disk.
"""
import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]

    df = pd.read_csv(data_cfg["processed_path"])
    target = data_cfg["target_column"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y
    )

    model = RandomForestClassifier(**model_cfg["params"])
    model.fit(X_train, y_train)

    Path(model_cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_cfg["save_path"])

    # Save test split for evaluation step
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"Model trained and saved to {model_cfg['save_path']}")


if __name__ == "__main__":
    main()
