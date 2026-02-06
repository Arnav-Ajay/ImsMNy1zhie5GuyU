import time
from collections import defaultdict

metrics = defaultdict(int)

def record_request(latency_ms: float):
    metrics["requests_total"] += 1
    metrics["latency_ms_sum"] += latency_ms

def snapshot():
    if metrics["requests_total"] == 0:
        avg_latency = 0
    else:
        avg_latency = metrics["latency_ms_sum"] / metrics["requests_total"]

    return {
        "requests_total": metrics["requests_total"],
        "avg_latency_ms": round(avg_latency, 2),
    }
