"""
Visualisation utilities.
Requires: pip install matplotlib seaborn scikit-learn
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.config import FIGURES  # Path object pointing to the figures/ output directory


def plot_class_distribution(df, label_col: str = "label", title: str = "Class Distribution",
                            filename: str = "class_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    # Count samples per class and plot as a bar chart
    df[label_col].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    _save(fig, filename)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str = "model"):
    # Compute the N×N confusion matrix from true and predicted labels
    cm  = confusion_matrix(y_true, y_pred)
    # Scale figure size based on number of classes so labels never overlap
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    # fmt="d" displays raw integer counts; cmap="Blues" gives the familiar blue heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    # Saved as confusion_matrix_<modelname>.png in figures/
    _save(fig, f"confusion_matrix_{model_name.lower()}.png")


def plot_roc_curves(roc_data: list):
    """
    roc_data: list of (fpr, tpr, label) tuples.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for fpr, tpr, label in roc_data:
        # Compute AUC directly from the fpr/tpr arrays using the trapezoidal rule
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    # Diagonal dashed line represents a random classifier (AUC=0.5); anything above is meaningful
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "roc_curves.png")


def plot_feature_importance(importances: dict, top_n: int = 20, model_name: str = "RF"):
    # Sort features by importance descending and take the top N
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals  = zip(*sorted_items)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Reverse slices so highest importance appears at the top of the horizontal bar chart
    ax.barh(names[::-1], vals[::-1], color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save(fig, "feature_importance.png")


def plot_amino_acid_frequency(df, label_col: str = "label", seq_col: str = "sequence",
                              filename: str = "aa_frequency.png"):
    """Bar chart of per-class amino acid composition (Gao et al. 2025 Fig 1 style)."""
    import collections
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # The 20 standard amino acids
    classes = sorted(df[label_col].unique())
    x = np.arange(len(AMINO_ACIDS))
    # Divide bar width evenly across all classes so bars don't overlap
    width = 0.8 / max(len(classes), 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cls in enumerate(classes):
        # Concatenate all sequences for this class into one long string for counting
        seqs = df[df[label_col] == cls][seq_col].str.cat()
        counts = collections.Counter(seqs)
        total = sum(counts[aa] for aa in AMINO_ACIDS) or 1  # Avoid division by zero
        # Convert raw counts to relative frequency for each amino acid
        freqs = [counts[aa] / total for aa in AMINO_ACIDS]
        ax.bar(x + i * width, freqs, width, label=str(cls))

    # Centre x-tick labels under the group of bars for each amino acid
    ax.set_xticks(x + width * (len(classes) - 1) / 2)
    ax.set_xticklabels(AMINO_ACIDS)
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Relative Frequency")
    ax.set_title("Amino Acid Frequency by Class")
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)


def plot_sequence_length_distribution(df, label_col: str = "label", seq_col: str = "sequence",
                                      filename: str = "seq_length_distribution.png"):
    """Overlapping histograms of peptide lengths per class."""
    classes = sorted(df[label_col].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in classes:
        # Compute character length of each sequence string for this class
        lengths = df[df[label_col] == cls][seq_col].str.len()
        # alpha=0.6 makes bars semi-transparent so overlapping classes are still visible
        ax.hist(lengths, bins=30, alpha=0.6, label=str(cls), edgecolor="none")
    ax.set_xlabel("Sequence Length (aa)")
    ax.set_ylabel("Count")
    ax.set_title("Peptide Length Distribution by Class")
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)


def plot_model_comparison(results_list: list, metrics=("accuracy", "f1_macro", "auc_roc"),
                          filename: str = "model_comparison.png"):
    """Grouped bar chart comparing classifiers across multiple metrics."""
    models  = [r["model"] for r in results_list]
    # Filter to only metrics that exist in at least one results dict
    metrics = [m for m in metrics if any(m in r for r in results_list)]
    x       = np.arange(len(models))
    # Divide bar width evenly so metric groups don't overlap per model
    width   = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        # Use NaN as fallback if a model doesn't have this metric (e.g. no AUC for some)
        vals = [r.get(metric, float("nan")) for r in results_list]
        ax.bar(x + i * width, vals, width, label=metric)

    # Centre tick labels under each model's group of bars
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)  # All metrics are 0-1; slight headroom above 1.0
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)


def plot_tsne(X, y, class_names=None, title: str = "t-SNE Feature Space",
              filename: str = "tsne.png"):
    """PCA → t-SNE scatter coloured by class (Ferdous et al. 2024 style)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Cap at 1000 points — t-SNE is O(n²) and too slow on the full dataset
    max_pts = 1000
    if len(X) > max_pts:
        idx = np.random.choice(len(X), max_pts, replace=False)
        X, y = X[idx], y[idx]

    # PCA first: reduces noise and speeds up t-SNE significantly
    # n_components capped at 50, feature count, or sample count — whichever is smallest
    n_components = min(50, X.shape[1], X.shape[0])
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X)
    # t-SNE reduces the PCA output to 2D for plotting
    X_2d  = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pca)

    classes = np.unique(y)
    cmap    = plt.cm.get_cmap("tab10", len(classes))  # Distinct color per class
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        mask = y == cls
        label = class_names[cls] if class_names is not None and cls < len(class_names) else str(cls)
        # s=15: small point size; alpha=0.7: slight transparency for dense regions
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=15, alpha=0.7,
                   color=cmap(i), label=label)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2)  # Larger legend markers since scatter points are small
    plt.tight_layout()
    _save(fig, filename)


def _save(fig, filename: str):
    # Create figures/ directory if it doesn't already exist
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, dpi=150)  # 150 DPI: good quality without excessive file size
    plt.close(fig)              # Close figure to free memory after saving
    print(f"Figure saved → {path}")