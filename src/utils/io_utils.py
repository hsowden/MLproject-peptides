import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_features(csv_path: Path) -> np.ndarray:
    return pd.read_csv(csv_path).values


def load_labels(parsed_csv: Path) -> np.ndarray:
    return pd.read_csv(parsed_csv)["label"].values
