"""
Visualisation utilities.
Requires: pip install matplotlib seaborn scikit-learn
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.config import FIGURES


def plot_class_distribution(df, label_col: str = "label", title: str = "Class Distribution",
                            filename: str = "class_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    df[label_col].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    _save(fig, filename)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str = "model"):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    _save(fig, f"confusion_matrix_{model_name.lower()}.png")


def plot_roc_curves(roc_data: list):
    """
    roc_data: list of (fpr, tpr, label) tuples.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for fpr, tpr, label in roc_data:
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "roc_curves.png")


def plot_feature_importance(importances: dict, top_n: int = 20, model_name: str = "RF"):
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals  = zip(*sorted_items)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], vals[::-1], color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save(fig, "feature_importance.png")


def plot_amino_acid_frequency(df, label_col: str = "label", seq_col: str = "sequence",
                              filename: str = "aa_frequency.png"):
    """Bar chart of per-class amino acid composition (Gao et al. 2025 Fig 1 style)."""
    import collections
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    classes = sorted(df[label_col].unique())
    x = np.arange(len(AMINO_ACIDS))
    width = 0.8 / max(len(classes), 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cls in enumerate(classes):
        seqs = df[df[label_col] == cls][seq_col].str.cat()
        counts = collections.Counter(seqs)
        total = sum(counts[aa] for aa in AMINO_ACIDS) or 1
        freqs = [counts[aa] / total for aa in AMINO_ACIDS]
        ax.bar(x + i * width, freqs, width, label=str(cls))

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
        lengths = df[df[label_col] == cls][seq_col].str.len()
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
    metrics = [m for m in metrics if any(m in r for r in results_list)]
    x       = np.arange(len(models))
    width   = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        vals = [r.get(metric, float("nan")) for r in results_list]
        ax.bar(x + i * width, vals, width, label=metric)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
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

    # Subsample to keep runtime manageable
    max_pts = 1000
    if len(X) > max_pts:
        idx = np.random.choice(len(X), max_pts, replace=False)
        X, y = X[idx], y[idx]

    # PCA pre-reduction
    n_components = min(50, X.shape[1], X.shape[0])
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X)
    X_2d  = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pca)

    classes = np.unique(y)
    cmap    = plt.cm.get_cmap("tab10", len(classes))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        mask = y == cls
        label = class_names[cls] if class_names is not None and cls < len(class_names) else str(cls)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=15, alpha=0.7,
                   color=cmap(i), label=label)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2)
    plt.tight_layout()
    _save(fig, filename)


def _save(fig, filename: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {path}")
