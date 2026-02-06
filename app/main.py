import time
import pandas as pd
from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import bundle
from app.monitoring import record_request, snapshot

app = FastAPI(title="Happiness ML Service")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": bundle.meta.get("model_type"),
        "features": bundle.meta.get("features"),
        "cv_accuracy": bundle.meta.get("cv_accuracy"),
        "holdout_accuracy": bundle.meta.get("holdout_accuracy"),
        "model_version": bundle.meta.get("trained_at"),
    }

@app.get("/metrics")
def metrics():
    return snapshot()

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.time()

    X = pd.DataFrame([[req.X1, req.X5]], columns=bundle.meta["features"])

    prob = bundle.model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    latency_ms = (time.time() - start) * 1000
    record_request(latency_ms)

    return PredictionResponse(
        prediction=pred,
        probability=round(prob, 4),
        model_version=bundle.meta.get("trained_at"),
        features_used=bundle.meta.get("features"),
    )
