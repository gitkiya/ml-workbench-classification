from __future__ import annotations
import pandas as pd

def validate_classification_data(df: pd.DataFrame, target: str, features: list[str]) -> None:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' does not exist.")
    if not features:
        raise ValueError("Choose at least one input feature.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown feature columns: {missing}")
    if target in features:
        raise ValueError("The target cannot also be an input feature.")
    if df[target].isna().any():
        raise ValueError("The target column contains missing values. Remove or resolve those rows first.")
    if df[target].nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if df[target].nunique() > 50:
        raise ValueError("This workbench currently limits classification targets to 50 distinct classes.")
    if len(df) < 10:
        raise ValueError("At least 10 rows are required for a meaningful train/test workflow.")
