import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/model.joblib")
FEATURES = ["X1", "X5"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x1", type=int, required=True, help="Delivered on time (1-5)")
    parser.add_argument("--x5", type=int, required=True, help="Courier satisfaction (1-5)")

    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)

    row = pd.DataFrame([{"X1": args.x1, "X5": args.x5}])[FEATURES]

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0, 1]

    label = "happy" if pred == 1 else "unhappy"
    print(f"Prediction: {label} (P(happy)={proba:.3f})")


if __name__ == "__main__":
    main()
