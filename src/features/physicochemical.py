"""
Physicochemical feature extraction using the Peptides library
(or a manual AAIndex-based fallback).
"""
import numpy as np
import pandas as pd
from src.config import TRAIN_PARSED, TEST_PARSED, TRAIN_PHYSCHEM, TEST_PHYSCHEM


# Per-residue physicochemical scales (AAIndex subset)
HYDROPHOBICITY = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
MOLECULAR_WEIGHT = {
    "A": 89,  "R": 174, "N": 132, "D": 133, "C": 121,
    "Q": 146, "E": 147, "G": 75,  "H": 155, "I": 131,
    "L": 131, "K": 146, "M": 149, "F": 165, "P": 115,
    "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117,
}
CHARGE_AT_PH7 = {
    "A": 0,  "R": 1,  "N": 0,  "D": -1, "C": 0,
    "Q": 0,  "E": -1, "G": 0,  "H": 0,  "I": 0,
    "L": 0,  "K": 1,  "M": 0,  "F": 0,  "P": 0,
    "S": 0,  "T": 0,  "W": 0,  "Y": 0,  "V": 0,
}


def _scale_stats(seq: str, scale: dict) -> dict:
    vals = [scale.get(aa, 0) for aa in seq]
    return {
        "mean": np.mean(vals),
        "std":  np.std(vals),
        "min":  np.min(vals),
        "max":  np.max(vals),
    }


def extract_physchem(sequence: str) -> dict:
    length = len(sequence)
    hydro  = _scale_stats(sequence, HYDROPHOBICITY)
    mw     = _scale_stats(sequence, MOLECULAR_WEIGHT)
    charge = _scale_stats(sequence, CHARGE_AT_PH7)
    net_charge = sum(CHARGE_AT_PH7.get(aa, 0) for aa in sequence)

    return {
        "length":          length,
        "net_charge":      net_charge,
        "hydro_mean":      hydro["mean"],
        "hydro_std":       hydro["std"],
        "hydro_min":       hydro["min"],
        "hydro_max":       hydro["max"],
        "mw_mean":         mw["mean"],
        "mw_std":          mw["std"],
        "charge_mean":     charge["mean"],
        "charge_std":      charge["std"],
        "frac_positive":   net_charge / length if length else 0,
        "frac_aromatic":   sum(1 for aa in sequence if aa in "FWY") / length if length else 0,
        "frac_polar":      sum(1 for aa in sequence if aa in "STNQ") / length if length else 0,
    }


def build_physchem_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = df["sequence"].apply(extract_physchem).apply(pd.Series)
    return feats


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PARSED)
    test  = pd.read_csv(TEST_PARSED)

    train_feats = build_physchem_features(train)
    test_feats  = build_physchem_features(test)

    train_feats.to_csv(TRAIN_PHYSCHEM, index=False)
    test_feats.to_csv(TEST_PHYSCHEM,  index=False)
    print(f"Physicochemical features: {train_feats.shape[1]} columns")
    print(f"Train → {TRAIN_PHYSCHEM}")
    print(f"Test  → {TEST_PHYSCHEM}")
