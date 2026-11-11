from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import build_preprocessor
from src.models.registry import get_classifiers, get_parameter_grids
from src.evaluation.metrics import calculate_metrics, make_confusion_matrix, make_classification_report

def train_and_compare(
    df: pd.DataFrame, target: str, features: list[str],
    test_size: float = 0.2, cv_folds: int = 5, tune: bool = False, random_state: int = 42
) -> dict:
    X = df[features].copy()
    y = df[target].copy()
    class_counts = y.value_counts()
    if class_counts.min() < cv_folds:
        raise ValueError(
            f"The smallest class has {class_counts.min()} row(s), fewer than {cv_folds} CV folds. "
            "Choose fewer folds or collect more examples."
        )

    # Split before fitting preprocessing: the test set cannot influence imputation/scaling/encoding.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_counts = y_train.value_counts()
    if train_counts.min() < cv_folds:
        raise ValueError("Training split has too few examples in at least one class for the chosen CV folds.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    classifiers = get_classifiers(random_state)
    grids = get_parameter_grids()

    rows, candidates = [], {}
    for name, estimator in classifiers.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        if tune:
            search = GridSearchCV(
                pipeline, grids[name], scoring="f1_weighted",
                cv=cv, n_jobs=-1, refit=True, error_score="raise"
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = search.best_params_
            cv_score = float(search.best_score_)
        else:
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
            cv_score = float(scores.mean())
            pipeline.fit(X_train, y_train)
            fitted = pipeline
            best_params = {}

        candidates[name] = {"pipeline": fitted, "cv_f1_weighted": cv_score, "best_params": best_params}
        rows.append({"model": name, "cv_f1_weighted_mean": cv_score, "best_params": str(best_params)})

    comparison = pd.DataFrame(rows).sort_values("cv_f1_weighted_mean", ascending=False).reset_index(drop=True)
    selected_name = comparison.iloc[0]["model"]
    selected_pipeline = candidates[selected_name]["pipeline"]

    # Test data is used only after model selection, for final unbiased evaluation.
    predictions = selected_pipeline.predict(X_test)
    labels = list(selected_pipeline.named_steps["model"].classes_)
    positive_label = labels[1] if len(labels) == 2 else None
    metrics = calculate_metrics(y_test, predictions, labels, positive_label)
    test_metrics = pd.DataFrame([{"model": selected_name, **metrics}])
    return {
        "selected_model": selected_name,
        "selection_metric": "weighted F1",
        "pipeline": selected_pipeline,
        "features": features,
        "target": target,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "confusion_matrix": make_confusion_matrix(y_test, predictions, labels),
        "classification_report": make_classification_report(y_test, predictions),
        "best_params": candidates[selected_name]["best_params"],
        "labels": labels,
        "test_size": test_size,
        "cv_folds": cv_folds,
    }
