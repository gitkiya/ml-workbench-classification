from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "models"
REPORT_DIR = ROOT / "artifacts" / "reports"
LATEST_MODEL = MODEL_DIR / "latest_model.joblib"
LATEST_METADATA = MODEL_DIR / "latest_metadata.json"

def save_experiment(result: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "selected_model": result["selected_model"],
        "features": result["features"],
        "target": result["target"],
        "labels": [str(x) for x in result["labels"]],
        "selection_metric": result["selection_metric"],
        "test_size": result["test_size"],
        "cv_folds": result["cv_folds"],
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(result["pipeline"], LATEST_MODEL)
    LATEST_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {
        "metadata": metadata,
        "comparison": result["comparison"].to_dict(orient="records"),
        "test_metrics": result["test_metrics"].to_dict(orient="records"),
        "confusion_matrix": result["confusion_matrix"].to_dict(),
    }
    report_path = REPORT_DIR / "latest_experiment.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return str(LATEST_MODEL), str(report_path)

def load_model_bundle():
    if not LATEST_MODEL.exists() or not LATEST_METADATA.exists():
        return None
    pipeline = joblib.load(LATEST_MODEL)
    metadata = json.loads(LATEST_METADATA.read_text(encoding="utf-8"))
    return {"pipeline": pipeline, "metadata": metadata}
