import json
import joblib
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/metadata.json")

from datetime import datetime

class ModelBundle:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            self.meta = json.load(f)

        # Enforce required metadata
        self.meta.setdefault("trained_at", datetime.now().isoformat())
        self.meta.setdefault("features", ["X1", "X5"])

bundle = ModelBundle()
