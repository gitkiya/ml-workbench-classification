from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def calculate_metrics(y_true, y_pred, labels, positive_label=None) -> dict:
    """Calculate metrics with weighted averaging for multiclass targets."""
    average = "binary" if len(labels) == 2 and positive_label is not None else "weighted"
    kwargs = {"average": average, "zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = positive_label
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, **kwargs)),
        "f1": float(f1_score(y_true, y_pred, **kwargs)),
    }

def make_confusion_matrix(y_true, y_pred, labels) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels],
    )

def make_classification_report(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()
