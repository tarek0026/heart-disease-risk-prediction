import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_data(path):
    """
    Load Heart Disease dataset and return X, y
    """

    df = pd.read_csv(path)

    # Rename target column
    df.rename(columns={"num": "target"}, inplace=True)

    # Convert target to binary
    df["target"] = (df["target"] > 0).astype(int)

    # Drop useless columns
    df.drop(["id", "dataset"], axis=1, inplace=True, errors="ignore")

    # Split features and label
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


def get_columns():
    """
    Return numerical and categorical columns
    """

    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]

    cat_cols = ["sex", "cp", "fbs", "restecg",
                "exang", "slope", "ca", "thal"]

    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    """
    Build preprocessing pipeline
    """

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocessor
