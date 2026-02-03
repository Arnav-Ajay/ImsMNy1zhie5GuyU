# Customer Happiness Prediction (Private Repo Name: ImsMNy1zhie5GuyU)

Goal: Predict whether a customer is happy (Y=1) or unhappy (Y=0) from survey answers X1–X6.

## Dataset
- `data/ACME-HappinessSurvey2020.csv`
- Target: `Y`
- Features: `X1..X6` (Likert 1–5)

## Approach
1. Train baseline models with Stratified 5-Fold CV (small dataset).
2. Run feature selection using exhaustive subset search (only 64 subsets).
3. Use the minimal feature set that preserves predictability.

**Result (feature selection):**
The strongest minimal feature set found is typically:
- `X1` (delivered on time)
- `X6` (app makes ordering easy)

## Setup
```bash
pip install -r requirements.txt
