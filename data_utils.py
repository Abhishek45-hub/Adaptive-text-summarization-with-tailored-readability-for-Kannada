import os
import pandas as pd
from datasets import Dataset

REQUIRED_COLS = {"id", "kannada_article", "kannada_highlights"}

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if not REQUIRED_COLS.issubset(df.columns):
        raise ValueError(f"CSV must contain {REQUIRED_COLS}")

    df = df.dropna(subset=["kannada_article", "kannada_highlights"])
    df = df[df["kannada_article"].str.strip() != ""]
    df = df[df["kannada_highlights"].str.strip() != ""]
    return df.reset_index(drop=True)

def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["input_text", "labels_text"]])
