# ML Workbench — Classification Edition

A modular end-to-end learning project for tabular **classification**. It is not just a Streamlit page: the interface in `app.py` calls a separate Python ML engine in `src/`.

## What it does

- Loads a CSV and profiles its columns, missing values, and basic statistics.
- Lets the user choose a target and input features.
- Splits data into train/test sets with stratification.
- Builds preprocessing inside a scikit-learn `Pipeline`:
  - numeric: median imputation + standardization
  - categorical: most-frequent imputation + one-hot encoding
- Compares Logistic Regression, Decision Tree, and Random Forest.
- Uses stratified cross-validation on the training partition to select the model by weighted F1.
- Optionally runs a small GridSearchCV for each model.
- Evaluates the selected model on the held-out test partition.
- Shows accuracy, precision, recall, F1, confusion matrix, and classification report.
- Saves the fitted preprocessing + estimator pipeline and experiment metadata.
- Loads a saved pipeline to predict new CSV rows.

## Architecture

```text
CSV
 │
 ▼
app.py (Streamlit UI)
 │
 ├── src/data/             loading, validation, profiling
 ├── src/preprocessing/    imputation, scaling, encoding
 ├── src/models/           classifier registry and search grids
 ├── src/training/         split, cross-validation, fitting, selection
 ├── src/evaluation/       metrics and confusion matrix
 ├── src/prediction/       prediction on raw new rows
 └── src/persistence/      save/load fitted pipeline and report
          │
          ▼
    artifacts/
      models/
      reports/
```

## Run locally

Python 3.10+ recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, upload a classification CSV, choose the target and features, and train.

## Run tests

```bash
python -m pytest -v
```

## Use with Titanic

For the common Kaggle Titanic training CSV, choose `Survived` as the target. Do not include `Survived` as a feature. Review columns such as `PassengerId`, `Name`, `Ticket`, and `Cabin` before including them; some are identifiers or high-cardinality text and may not help a first baseline. This app intentionally does not silently drop columns because feature selection is part of the experiment.

## Important design decisions

### Preventing data leakage
The train/test split happens before any preprocessing is fitted. The imputer, scaler, and encoder live inside the model pipeline. During cross-validation, each fold fits its own preprocessing only on that fold's training portion. The held-out test set is not used for model selection.

### Why cross-validation first?
The models are compared using weighted F1 across folds of the training data. The highest mean CV score selects the candidate. Only then is the chosen fitted pipeline evaluated on the test set. This avoids choosing a model based on test-set performance.

### Why weighted F1?
The default is a practical general-purpose metric when classes may be imbalanced or there may be more than two classes. For a real use case, choose the metric based on the cost of false positives and false negatives. The metric is a selection rule, not a universal definition of the best model.

### Scaling
Numeric columns are standardized in this baseline pipeline. This is important for Logistic Regression and usually harmless for tree models, though it is not required by trees.

### Limitations / next improvements
This is a complete first learning version, not a production ML platform. Future extensions could include configurable positive-class choice, threshold tuning, richer EDA, feature importance, model versioning, data drift checks, authentication, and deployment. Do not use it for consequential decisions without domain-specific validation and safeguards.
