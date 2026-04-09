from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import evaluate
from src.config import RANDOM_SEED


def build_knn(n_neighbors: int = 5, metric: str = "euclidean") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)),
    ])


def train_knn(X_train, y_train, X_test, y_test, **kwargs) -> dict:
    model = build_knn(**kwargs)
    model.fit(X_train, y_train)
    results = evaluate(model, X_test, y_test, model_name="KNN")
    return results, model
