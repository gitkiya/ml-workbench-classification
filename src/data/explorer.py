from __future__ import annotations
import pandas as pd

def dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
    }).rename_axis("column").reset_index()
