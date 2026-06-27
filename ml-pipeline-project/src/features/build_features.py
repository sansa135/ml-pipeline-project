"""
build_features.py
Feature engineering: encoding categorical variables, scaling, creating new features.
"""
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def main():
    config = load_config()
    processed_path = config["data"]["processed_path"]

    df = pd.read_csv(processed_path)
    df = encode_categoricals(df)

    df.to_csv(processed_path, index=False)
    print(f"Features built and saved to {processed_path}")


if __name__ == "__main__":
    main()
