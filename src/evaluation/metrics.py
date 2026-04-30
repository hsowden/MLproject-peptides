import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None, model_name: str = "") -> dict:
    # Build results dict with core classification metrics for any model
    results = {
        "model":     model_name,
        "accuracy":  accuracy_score(y_true, y_pred),
        # Macro: each class weighted equally regardless of sample count
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        # Weighted: each class weighted by its proportion in the dataset
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        # MCC: best single metric for imbalanced classes; ranges from -1 (worst) to +1 (best)
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                # Binary case: use probability of the positive class (column index 1)
                results["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multiclass case: one-vs-rest AUC averaged across all classes
                results["auc_roc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except ValueError:
            # AUC undefined if only one class present in y_true (e.g. small test splits)
            results["auc_roc"] = float("nan")
    return results


def evaluate(model, X_test, y_test, model_name: str = "") -> dict:
    # Used by classical sklearn models (KNN, SVM, RF, etc.) via their .predict() interface
    # CNN bypasses this and calls compute_metrics() directly after its own inference
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        # Not all sklearn models support probability output (e.g. linear SVM doesn't by default)
        y_prob = model.predict_proba(X_test)
    results = compute_metrics(y_test, y_pred, y_prob, model_name=model_name)
    # Attach raw arrays so plots.py can generate confusion matrices and ROC curves
    # These underscore keys are stripped before saving to CSV
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
            continue  # Already printed as header, skip here
        # Format floats to 4 decimal places; print other types (arrays, strings) as-is
        print(f"  {k:<14}: {v:.4f}" if isinstance(v, float) else f"  {k:<14}: {v}")