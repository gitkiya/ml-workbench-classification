from __future__ import annotations
import pandas as pd

def load_csv(file_or_path) -> pd.DataFrame:
    """Read a CSV into a DataFrame and reject empty or duplicate-column datasets."""
    df = pd.read_csv(file_or_path)
    if df.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")
    return df
