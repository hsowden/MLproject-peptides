from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import evaluate


def build_nb() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("nb",     GaussianNB()),
    ])


def train_nb(X_train, y_train, X_test, y_test) -> dict:
    model = build_nb()
    model.fit(X_train, y_train)
    results = evaluate(model, X_test, y_test, model_name="NaiveBayes")
    return results, model
