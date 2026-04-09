import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None, model_name: str = "") -> dict:
    results = {
        "model":     model_name,
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                results["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                results["auc_roc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except ValueError:
            results["auc_roc"] = float("nan")
    return results


def evaluate(model, X_test, y_test, model_name: str = "") -> dict:
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    results = compute_metrics(y_test, y_pred, y_prob, model_name=model_name)
    # Store raw arrays for plotting (stripped before CSV save)
    results["_y_true"] = y_test
    results["_y_pred"] = y_pred
    results["_y_prob"] = y_prob
    return results


def print_metrics(results: dict) -> None:
    print(f"\n{'='*40}")
    print(f"  {results.get('model', 'Model')}")
    print(f"{'='*40}")
    for k, v in results.items():
        if k == "model":
            continue
        print(f"  {k:<14}: {v:.4f}" if isinstance(v, float) else f"  {k:<14}: {v}")
