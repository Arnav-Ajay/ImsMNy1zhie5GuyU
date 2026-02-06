from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_predict():
    payload = {"X1": 3, "X5": 5}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
