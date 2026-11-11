import pandas as pd
from src.training.workflow import train_and_compare
from src.prediction.service import predict_rows

def sample_data():
    return pd.DataFrame({
        "age": [18, 22, 25, 29, 31, 35, 40, 45, 50, 55, 60, 65, 20, 28, 38, 48, 58, 68, 23, 43, 53, 63, 33, 73],
        "fare": [7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 6, 11, 18, 22, 32, 42, 9, 21, 33, 44, 14, 50],
        "gender": ["f", "m", "f", "m", "f", "m", "f", "m", "f", "m", "f", "m"] * 2,
        "survived": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })

def test_training_and_prediction():
    df = sample_data()
    result = train_and_compare(df, "survived", ["age", "fare", "gender"], cv_folds=3)
    assert result["selected_model"] in {"Logistic Regression", "Decision Tree", "Random Forest"}
    assert len(result["comparison"]) == 3
    prediction = predict_rows(result["pipeline"], pd.DataFrame({"age": [26], "fare": [13], "gender": ["f"]}), ["age", "fare", "gender"])
    assert "predicted_class" in prediction.columns
    assert len(prediction) == 1
