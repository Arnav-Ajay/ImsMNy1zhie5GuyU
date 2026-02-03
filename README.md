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


## Setup
```bash
pip install -r requirements.txt

