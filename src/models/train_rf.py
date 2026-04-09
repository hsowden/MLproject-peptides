from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.evaluation.metrics import evaluate
from src.config import RANDOM_SEED


def build_rf(n_estimators: int = 200, max_depth=None) -> Pipeline:
    return Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])


def train_rf(X_train, y_train, X_test, y_test, **kwargs) -> dict:
    model = build_rf(**kwargs)
    model.fit(X_train, y_train)
    results = evaluate(model, X_test, y_test, model_name="RandomForest")
    return results, model


def get_feature_importances(model: Pipeline, feature_names: list) -> dict:
    rf = model.named_steps["rf"]
    return dict(zip(feature_names, rf.feature_importances_))
