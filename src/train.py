import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = Path("data/ACME-HappinessSurvey2020.csv")
MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/metadata.json")

FEATURES = ["X1", "X5"]
TARGET = "Y"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    )

    # -----------------------
    # Cross-validated accuracy
    # -----------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # -----------------------
    # Hold-out accuracy (single split)
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    holdout_accuracy = accuracy_score(y_test, y_pred)

    # -----------------------
    # Fit final model on full data
    # -----------------------
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "model": "DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, min_samples_split=2, random_state=42)",

        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_scores": [float(s) for s in cv_scores],

        "holdout_accuracy": float(holdout_accuracy),
        "holdout_split": "80/20 stratified",
        "random_state": 42,
    }

    META_PATH.write_text(json.dumps(metadata, indent=2))

    print("Saved:", MODEL_PATH)
    print(f"CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Hold-out accuracy (80/20): {holdout_accuracy:.4f}")


if __name__ == "__main__":
    main()
