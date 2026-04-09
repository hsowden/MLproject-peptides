from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import evaluate
from src.config import RANDOM_SEED


def build_svm(C: float = 1.0, kernel: str = "rbf", probability: bool = True) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(C=C, kernel=kernel, probability=probability,
                       random_state=RANDOM_SEED, class_weight="balanced")),
    ])


def train_svm(X_train, y_train, X_test, y_test, **kwargs) -> dict:
    model = build_svm(**kwargs)
    model.fit(X_train, y_train)
    results = evaluate(model, X_test, y_test, model_name="SVM")
    return results, model
