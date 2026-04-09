import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from src.config import CV_FOLDS, RANDOM_SEED


def cross_validate_model(model_builder, X, y, n_splits: int = CV_FOLDS) -> pd.DataFrame:
    """
    Stratified k-fold CV.

    model_builder: callable that returns a fresh unfitted model/pipeline.
    Returns a DataFrame with per-fold metrics.
    """
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rows   = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_builder()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        rows.append({
            "fold":      fold,
            "accuracy":  accuracy_score(y_val, y_pred),
            "f1_macro":  f1_score(y_val, y_pred, average="macro", zero_division=0),
            "mcc":       matthews_corrcoef(y_val, y_pred),
        })
        print(f"  Fold {fold}: acc={rows[-1]['accuracy']:.3f}  f1={rows[-1]['f1_macro']:.3f}  mcc={rows[-1]['mcc']:.3f}")

    df = pd.DataFrame(rows)
    print(f"\n  Mean ± Std")
    for col in ["accuracy", "f1_macro", "mcc"]:
        print(f"    {col:<10}: {df[col].mean():.4f} ± {df[col].std():.4f}")
    return df
