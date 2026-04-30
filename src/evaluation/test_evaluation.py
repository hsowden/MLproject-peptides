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
    rows = []      # Accumulates cleaned metric dicts for final CSV
    roc_data = []  # Accumulates (fpr, tpr, model_name) tuples for the combined ROC plot

    for results, _ in models_and_data:
        # Print formatted metrics table to console for this model
        print_metrics(results)

        # Pop raw arrays out of the dict — needed for plots but must not go into the CSV
        y_true = results.pop("_y_true", None)
        y_pred = results.pop("_y_pred", None)
        y_prob = results.pop("_y_prob", None)
        model_name = results.get("model", "model")

        if y_true is not None and y_pred is not None:
            # Use provided class names or fall back to sorted unique label integers
            names = list(class_names) if class_names is not None \
                    else [str(i) for i in sorted(set(y_true))]
            # Generate and save confusion_matrix_<ModelName>.png
            plot_confusion_matrix(y_true, y_pred, names, model_name=model_name)

        # ROC curve only valid for binary classifiers with probability output
        if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
            # Compute false positive rate and true positive rate at all thresholds
            # _ discards the threshold values — not needed for plotting
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_data.append((fpr, tpr, model_name))

        # Append cleaned results dict (no raw arrays) as one row in the final table
        rows.append(results)

    if roc_data:
        # Plot all models' ROC curves overlaid on one figure with AUC labels
        plot_roc_curves(roc_data)

    # Build summary DataFrame indexed by model name
    df = pd.DataFrame(rows).set_index("model")
    out = TABLES / "classical_model_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)  # Create tables/ dir if it doesn't exist
    df.to_csv(out)
    print(f"\nResults saved → {out}")
    return df