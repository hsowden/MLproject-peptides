"""
Sequence-based feature extraction: amino acid composition, dipeptide composition,
k-mer counts, and position-specific features.
"""
import numpy as np
import pandas as pd
from itertools import product
from src.config import TRAIN_PARSED, TEST_PARSED, TRAIN_SEQ, TEST_SEQ

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def amino_acid_composition(sequence: str) -> dict:
    length = len(sequence)
    return {f"aac_{aa}": sequence.count(aa) / length if length else 0 for aa in AMINO_ACIDS}


def dipeptide_composition(sequence: str) -> dict:
    length = len(sequence)
    pairs = ["".join(p) for p in product(AMINO_ACIDS, repeat=2)]
    counts = {f"dpc_{p}": 0 for p in pairs}
    for i in range(len(sequence) - 1):
        key = f"dpc_{sequence[i:i+2]}"
        if key in counts:
            counts[key] += 1
    total = length - 1 if length > 1 else 1
    return {k: v / total for k, v in counts.items()}


def ctd_composition(sequence: str) -> dict:
    """Composition/Transition/Distribution — hydrophobicity grouping."""
    groups = {
        "polar":       set("RKEDQN"),
        "neutral":     set("GASTPHY"),
        "hydrophobic": set("CVLIMFW"),
    }
    length = len(sequence)
    feats = {}
    for name, aas in groups.items():
        feats[f"ctd_comp_{name}"] = sum(1 for aa in sequence if aa in aas) / length if length else 0
    return feats


def build_sequence_features(df: pd.DataFrame, include_dipeptide: bool = True) -> pd.DataFrame:
    aac  = df["sequence"].apply(amino_acid_composition).apply(pd.Series)
    ctd  = df["sequence"].apply(ctd_composition).apply(pd.Series)
    if include_dipeptide:
        dpc = df["sequence"].apply(dipeptide_composition).apply(pd.Series)
        return pd.concat([aac, dpc, ctd], axis=1)
    return pd.concat([aac, ctd], axis=1)


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PARSED)
    test  = pd.read_csv(TEST_PARSED)

    train_feats = build_sequence_features(train)
    test_feats  = build_sequence_features(test)

    train_feats.to_csv(TRAIN_SEQ, index=False)
    test_feats.to_csv(TEST_SEQ,  index=False)
    print(f"Sequence features: {train_feats.shape[1]} columns")
    print(f"Train → {TRAIN_SEQ}")
    print(f"Test  → {TEST_SEQ}")
