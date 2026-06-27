"""
make_dataset.py
Loads raw data and performs basic cleaning, then saves to processed folder.
"""
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows where target is missing
    df = df.dropna(subset=[df.columns[-1]])

    # Fill numeric NaNs with median, categorical NaNs with mode
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    config = load_config()
    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    df_clean = clean_data(df)
    df_clean.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")


if __name__ == "__main__":
    main()
