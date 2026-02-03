import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

model = joblib.load("models/model.joblib")

FEATURES = ["X1", "X5"]

plt.figure(figsize=(14, 6))
plot_tree(
    model,
    feature_names=FEATURES,
    class_names=["Unhappy", "Happy"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Customer Happiness")
plt.show()
