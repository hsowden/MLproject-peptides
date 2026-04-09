"""
Combine physicochemical, sequence-based, and PLM features into a single fused matrix.
"""
import numpy as np
import pandas as pd
from src.config import (
    TRAIN_PHYSCHEM, TEST_PHYSCHEM,
    TRAIN_SEQ, TEST_SEQ,
    TRAIN_PLM, TEST_PLM,
    TRAIN_FUSED, TEST_FUSED,
)


def fuse(physchem_path, seq_path, plm_path) -> pd.DataFrame:
    physchem = pd.read_csv(physchem_path)
    seq      = pd.read_csv(seq_path)
    plm_arr  = np.load(plm_path)
    plm_cols = [f"plm_{i}" for i in range(plm_arr.shape[1])]
    plm_df   = pd.DataFrame(plm_arr, columns=plm_cols)
    return pd.concat([physchem, seq, plm_df], axis=1)


if __name__ == "__main__":
    train_fused = fuse(TRAIN_PHYSCHEM, TRAIN_SEQ, TRAIN_PLM)
    test_fused  = fuse(TEST_PHYSCHEM,  TEST_SEQ,  TEST_PLM)

    train_fused.to_csv(TRAIN_FUSED, index=False)
    test_fused.to_csv(TEST_FUSED,  index=False)
    print(f"Fused features: {train_fused.shape[1]} columns")
    print(f"Train → {TRAIN_FUSED}")
    print(f"Test  → {TEST_FUSED}")
