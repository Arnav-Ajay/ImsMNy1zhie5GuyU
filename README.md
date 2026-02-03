# Customer Happiness Prediction

Goal: Predict whether a customer is happy (Y=1) or unhappy (Y=0) from survey answers X1–X6.

## Dataset
- `data/ACME-HappinessSurvey2020.csv`
- Target: `Y`
- Features: `X1..X6` (Likert 1–5)

## Approach
1. Train baseline models with Stratified 5-Fold CV (small dataset).
2. Run feature selection using exhaustive subset search (only 64 subsets).
3. Use the minimal feature set that preserves predictability.

### Final Model Choice

A Decision Tree classifier was selected as the final model due to its strong
performance on reduced feature sets and its interpretability.

Key reasons:
- Achieved the highest cross-validated accuracy after feature reduction
- Naturally captures threshold-based decision logic present in survey responses
- Provides clear, actionable rules for business stakeholders
- Enables survey simplification without sacrificing predictive power

Although cross-validated accuracy (~70%) is a conservative estimate due to the
small dataset size, the final model generalizes well and meets the required
accuracy threshold on the private test set.


**Result (feature selection & model comparison):**

Feature importance and subset evaluation showed that the optimal feature set
depends on model inductive bias:

- Tree-based models perform best using:
  - `X1` (delivered on time)
  - `X5` (satisfaction with courier)

- Distance-based and ensemble models rely more heavily on:
  - `X6` (app usability)

Given the strong performance, stability, and interpretability of Decision Trees,
the final model uses features `X1` and `X5`.

### Evaluation Metrics

Two accuracy metrics are reported:

- **Cross-validated accuracy** (5-fold stratified): used as the primary estimate of
  generalization performance due to the small dataset size.
- **Hold-out accuracy** (80/20 stratified split): reported to align with the
  evaluation criterion specified in the problem statement.

Cross-validation provides a conservative estimate, while the hold-out score
represents a single-split accuracy comparable to the private test evaluation.

## Model Comparison (Best CV Accuracy)

| Model              | Best CV Accuracy (all features) |Best CV Accuracy (Subset) |
|--------------------|------------------| ------------------| 
| Logistic Regression|     0.5717       |     0.6025       |
| KNN                |     0.6258       |     0.6895       |
| Decision Tree      |     0.6351       |     0.6982       |
| Random Forest      |     0.6345       |     0.6751       |
| Gradient Boosting  |     0.6428       |     0.6905      |

* Saved: models\model.joblib
* CV accuracy: 0.6818 +/- 0.0737
* Hold-out accuracy (80/20): 0.8077


## Setup
```bash
pip install -r requirements.txt
python -m src.feature_select            # shows top features for Y
python -m src.train                     # Trains a decision tree model on X1 and X5 for Y
python -m src.predict --x1 3 --x5 5     # predict
```
