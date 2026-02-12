from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import joblib

from src.preprocess import load_data, build_preprocessor


def train_model():

    # Load data
    X, y = load_data("Data/heart_disease_uci.csv")

    # Columns
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    # Preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Model Pipeline
    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", HistGradientBoostingClassifier(random_state=42))
    ])

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Params
    params = {
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__max_depth": [3, 5, None],
        "clf__max_iter": [100, 200, 300],
        "clf__min_samples_leaf": [10, 20, 30]
    }

    # GridSearch
    grid = GridSearchCV(
        model,
        params,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X, y)

    print("Best ROC-AUC:", grid.best_score_)
    print("Best Params:", grid.best_params_)

    best_model = grid.best_estimator_

    # Save model
    joblib.dump(best_model, "Model/heart_disease_model.pkl")
    print("✅ Model saved to Model/heart_disease_model.pkl")

    return best_model


if __name__ == "__main__":
    train_model()
