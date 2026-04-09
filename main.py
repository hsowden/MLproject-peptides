"""
Peptide Toxicity Classification — main pipeline entry point.

Steps
-----
1. Parse FASTA files
2. Extract features (physicochemical, sequence-based, PLM)
3. Fuse features
4. Train classical models + CNN
5. Evaluate on test set
"""
import argparse
import pandas as pd
import numpy as np

from src.utils.seed import set_seed
from src.utils.io_utils import load_features, load_labels
from src.config import (
    TRAIN_FASTA, TEST_FASTA,
    TRAIN_PARSED, TEST_PARSED,
    TRAIN_PHYSCHEM, TEST_PHYSCHEM,
    TRAIN_SEQ, TEST_SEQ,
    TRAIN_PLM, TEST_PLM,
    TRAIN_FUSED, TEST_FUSED,
    TABLES,
)


def step_parse():
    from src.parser import parse_fasta, records_to_csv, build_class_summary
    from src.evaluation.plots import (plot_class_distribution,
                                      plot_amino_acid_frequency,
                                      plot_sequence_length_distribution)
    print("\n[1/5] Parsing FASTA files...")
    train = parse_fasta(TRAIN_FASTA)
    test  = parse_fasta(TEST_FASTA)
    records_to_csv(train, TRAIN_PARSED)
    records_to_csv(test,  TEST_PARSED)
    build_class_summary(train, test, TRAIN_PARSED.parent / "class_summary.csv")
    train_df = pd.DataFrame(train, columns=["id", "label", "sequence"])
    test_df  = pd.DataFrame(test,  columns=["id", "label", "sequence"])
    plot_class_distribution(train_df, label_col="label", title="Train Class Distribution",
                            filename="class_distribution_train.png")
    plot_class_distribution(test_df,  label_col="label", title="Test Class Distribution",
                            filename="class_distribution_test.png")
    plot_amino_acid_frequency(train_df, label_col="label", seq_col="sequence",
                              filename="aa_frequency_train.png")
    plot_sequence_length_distribution(train_df, label_col="label", seq_col="sequence",
                                      filename="seq_length_distribution_train.png")
    plot_sequence_length_distribution(test_df,  label_col="label", seq_col="sequence",
                                      filename="seq_length_distribution_test.png")


def step_features():
    from src.features.physicochemical import build_physchem_features
    from src.features.sequence_based import build_sequence_features
    from src.features.plm_embeddings import load_model, embed_sequences, get_device
    from src.features.feature_fusion import fuse
    import numpy as np

    train_df = pd.read_csv(TRAIN_PARSED)
    test_df  = pd.read_csv(TEST_PARSED)

    print("\n[2/5] Extracting physicochemical features...")
    build_physchem_features(train_df).to_csv(TRAIN_PHYSCHEM, index=False)
    build_physchem_features(test_df).to_csv(TEST_PHYSCHEM,  index=False)

    print("Extracting sequence-based features...")
    build_sequence_features(train_df).to_csv(TRAIN_SEQ, index=False)
    build_sequence_features(test_df).to_csv(TEST_SEQ,  index=False)

    print("Extracting PLM embeddings (may take a while)...")
    tokenizer, model = load_model()
    device = get_device()
    np.save(TRAIN_PLM, embed_sequences(train_df["sequence"].tolist(), tokenizer, model, device=device))
    np.save(TEST_PLM,  embed_sequences(test_df["sequence"].tolist(),  tokenizer, model, device=device))

    print("Fusing features...")
    fuse(TRAIN_PHYSCHEM, TRAIN_SEQ, TRAIN_PLM).to_csv(TRAIN_FUSED, index=False)
    fuse(TEST_PHYSCHEM,  TEST_SEQ,  TEST_PLM).to_csv(TEST_FUSED,  index=False)


def step_train_classical():
    from src.models.train_knn import train_knn
    from src.models.train_svm import train_svm
    from src.models.train_nb  import train_nb
    from src.models.train_rf  import train_rf, get_feature_importances
    from src.models.train_bagging import train_bagging
    from src.preprocessing import encode_labels
    from src.evaluation.test_evaluation import evaluate_all
    from src.evaluation.plots import plot_feature_importance, plot_tsne, plot_model_comparison

    print("\n[3/5] Training classical models...")
    train_df = pd.read_csv(TRAIN_PARSED)
    test_df  = pd.read_csv(TEST_PARSED)
    train_df, test_df, le = encode_labels(train_df, test_df)

    X_train = load_features(TRAIN_FUSED)
    X_test  = load_features(TEST_FUSED)
    y_train = train_df["label_enc"].values
    y_test  = test_df["label_enc"].values
    feature_names = list(pd.read_csv(TRAIN_FUSED).columns)

    plot_tsne(X_train, y_train, class_names=list(le.classes_),
              title="t-SNE of Fused Features (Train)", filename="tsne_train.png")

    all_results = []
    for fn in [train_knn, train_svm, train_nb, train_rf, train_bagging]:
        res, model = fn(X_train, y_train, X_test, y_test)
        all_results.append((res, model))
        # Feature importance for RF
        if fn is train_rf:
            importances = get_feature_importances(model, feature_names)
            plot_feature_importance(importances, top_n=20, model_name="RandomForest")

    evaluate_all(all_results, class_names=le.classes_)

    # Model comparison chart (raw arrays already popped by evaluate_all)
    clean_results = [res for res, _ in all_results]
    plot_model_comparison(clean_results, filename="model_comparison_classical.png")


def step_train_cnn():
    from src.models.train_cnn import train_cnn
    from src.evaluation.metrics import print_metrics
    from src.evaluation.plots import plot_confusion_matrix, plot_roc_curves
    from sklearn.metrics import roc_curve

    print("\n[4/5] Training CNN...")
    train_df = pd.read_csv(TRAIN_PARSED)
    test_df  = pd.read_csv(TEST_PARSED)

    results, model, le = train_cnn(
        train_df["sequence"].tolist(), train_df["label"].values,
        test_df["sequence"].tolist(),  test_df["label"].values,
    )
    print_metrics(results)

    y_true = results.pop("_y_true", None)
    y_pred = results.pop("_y_pred", None)
    y_prob = results.pop("_y_prob", None)

    if y_true is not None and y_pred is not None:
        plot_confusion_matrix(y_true, y_pred, list(le.classes_), model_name="CNN")

    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plot_roc_curves([(fpr, tpr, "CNN")])

    out = TABLES / "deep_model_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(out, index=False)
    print(f"CNN results → {out}")


def main():
    parser = argparse.ArgumentParser(description="Peptide Toxicity Pipeline")
    parser.add_argument("--steps", nargs="+",
                        choices=["parse", "features", "classical", "cnn", "all"],
                        default=["all"])
    args = parser.parse_args()
    steps = args.steps if "all" not in args.steps else ["parse", "features", "classical", "cnn"]

    set_seed()
    if "parse"     in steps: step_parse()
    if "features"  in steps: step_features()
    if "classical" in steps: step_train_classical()
    if "cnn"       in steps: step_train_cnn()
    print("\nDone.")


if __name__ == "__main__":
    main()
