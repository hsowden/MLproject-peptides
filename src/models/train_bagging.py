from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import evaluate
from src.config import RANDOM_SEED


def build_bagging(n_estimators: int = 100, max_samples: float = 0.8) -> Pipeline:
    base = DecisionTreeClassifier(random_state=RANDOM_SEED)
    return Pipeline([
        ("scaler",  StandardScaler()),
        ("bagging", BaggingClassifier(
            estimator=base,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])


def train_bagging(X_train, y_train, X_test, y_test, **kwargs) -> dict:
    model = build_bagging(**kwargs)
    model.fit(X_train, y_train)
    results = evaluate(model, X_test, y_test, model_name="Bagging")
    return results, model
