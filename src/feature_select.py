from itertools import combinations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_PATH = "data/ACME-HappinessSurvey2020.csv"
FEATURES_ALL = ["X1", "X2", "X3", "X4", "X5", "X6"]
TARGET = "Y"


def main():
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for r in range(1, len(FEATURES_ALL) + 1):
        for cols in combinations(FEATURES_ALL, r):
            X = df[list(cols)]
            model = DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=2,
                random_state=42
            )

            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            results.append((scores.mean(), scores.std(), cols))

    results.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 10 feature sets (by mean CV accuracy):")
    for mean_acc, std_acc, cols in results[:10]:
        print(f"  acc={mean_acc:.4f} +/- {std_acc:.4f} | {cols}")

    best = results[0]
    print("\nBEST:", best)


if __name__ == "__main__":
    main()
