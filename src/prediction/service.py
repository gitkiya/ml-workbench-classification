from __future__ import annotations
import pandas as pd

def predict_rows(pipeline, new_rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [column for column in features if column not in new_rows.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required feature columns: {missing}")
    X_new = new_rows[features].copy()
    predicted = pipeline.predict(X_new)
    output = new_rows.copy()
    output["predicted_class"] = predicted
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_new)
        classes = pipeline.named_steps["model"].classes_
        for index, class_label in enumerate(classes):
            output[f"probability_{class_label}"] = probabilities[:, index]
    return output
