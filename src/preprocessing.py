import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import TRAIN_PARSED, TEST_PARSED


VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def load_parsed(train_path=TRAIN_PARSED, test_path=TEST_PARSED):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


def filter_sequences(df: pd.DataFrame, max_len: int = 200) -> pd.DataFrame:
    """Remove sequences with non-standard amino acids or exceeding max_len."""
    mask_valid = df["sequence"].apply(lambda s: set(s).issubset(VALID_AAS))
    mask_len   = df["sequence"].str.len() <= max_len
    filtered = df[mask_valid & mask_len].reset_index(drop=True)
    print(f"Kept {len(filtered)}/{len(df)} sequences after filtering")
    return filtered


def encode_labels(train: pd.DataFrame, test: pd.DataFrame):
    le = LabelEncoder()
    train = train.copy()
    test  = test.copy()
    train["label_enc"] = le.fit_transform(train["label"])
    test["label_enc"]  = le.transform(test["label"])
    return train, test, le


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
