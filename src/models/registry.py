from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_classifiers(random_state: int = 42):
    """Return the baseline classification algorithms supported by this project."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced", n_jobs=-1
        ),
    }

def get_parameter_grids():
    return {
        "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
        "Decision Tree": {"model__max_depth": [None, 4, 8], "model__min_samples_leaf": [1, 3]},
        "Random Forest": {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 3]},
    }
