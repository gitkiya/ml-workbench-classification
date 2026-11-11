import pandas as pd
import pytest
from src.data.validator import validate_classification_data

def test_rejects_missing_target():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, None, 1]})
    with pytest.raises(ValueError):
        validate_classification_data(df, "y", ["x"])

def test_rejects_target_as_feature():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with pytest.raises(ValueError):
        validate_classification_data(df, "y", ["x", "y"])
