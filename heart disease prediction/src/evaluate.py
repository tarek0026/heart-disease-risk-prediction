import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import roc_curve, auc
from src.preprocess import load_data


def evaluate_model():
    # Load model
    model = joblib.load("Model/heart_disease_model.pkl")

    # Load data
    X, y = load_data("Data/heart_disease_uci.csv")

    # ============================
    # Split once (FAST)
    # ============================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Fit model once
    model.fit(X_train, y_train)

    # ============================
    # 1) ROC Curve
    # ============================

    y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Heart Disease Prediction")
    plt.legend()
    plt.show()

    # ============================
    # 2) Learning Curve (FAST version)
    # ============================

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X, y,
        cv=3,   # بدل 5 عشان السرعة
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, val_mean, label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("ROC-AUC")
    plt.title("Learning Curve - Heart Disease Model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
