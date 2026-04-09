"""
Run all trained models on the held-out test set and collect results.
"""
import pandas as pd
from sklearn.metrics import roc_curve
from src.evaluation.metrics import print_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curves
from src.config import TABLES


def evaluate_all(models_and_data: list, class_names=None) -> pd.DataFrame:
    """
    models_and_data: list of (results_dict, model_name) tuples
                     where results_dict comes from evaluate() or compute_metrics().
    class_names: list of string class labels (in encoded order).
    Saves a summary CSV, prints per-model metrics, and generates plots.
    """
    rows = []
    roc_data = []

    for results, _ in models_and_data:
        print_metrics(results)

        # Extract raw arrays stored by evaluate() — not saved to CSV
        y_true = results.pop("_y_true", None)
        y_pred = results.pop("_y_pred", None)
        y_prob = results.pop("_y_prob", None)
        model_name = results.get("model", "model")

        if y_true is not None and y_pred is not None:
            names = list(class_names) if class_names is not None \
                    else [str(i) for i in sorted(set(y_true))]
            plot_confusion_matrix(y_true, y_pred, names, model_name=model_name)

        # ROC curve — binary only
        if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_data.append((fpr, tpr, model_name))

        rows.append(results)

    if roc_data:
        plot_roc_curves(roc_data)

    df = pd.DataFrame(rows).set_index("model")
    out = TABLES / "classical_model_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\nResults saved → {out}")
    return df
