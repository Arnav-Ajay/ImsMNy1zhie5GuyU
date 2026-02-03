import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_PATH = Path("data/ACME-HappinessSurvey2020.csv")
MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/metadata.json")

# Minimal set (from feature selection results)
FEATURES = ["X1", "X5"]
TARGET = "Y"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    model = DecisionTreeClassifier(max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    )


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Fit final model on all available training data
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "cv_scores": [float(s) for s in scores],
        "model": "DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, min_samples_split=2, random_state=42)",
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))

    print("Saved:", MODEL_PATH)
    print("CV accuracy:", round(scores.mean(), 4), "+/-", round(scores.std(), 4))


if __name__ == "__main__":
    main()
