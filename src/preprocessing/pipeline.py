from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

def build_preprocessor(X):
    """Create numeric and categorical preprocessing based only on training feature dtypes."""
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def clone_pipeline(pipeline):
    """Return an unfitted copy of a sklearn pipeline."""
    return clone(pipeline)
