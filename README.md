# Instagram Usage Modeling — RQ2

Predicting daily Instagram usage time from demographic, lifestyle, and
well-being variables. Data science capstone, Fall 2025.

## Research question

What demographic, lifestyle, and well-being factors predict the number of
minutes a user spends on Instagram per day?

## Data

`instagram_users_lifestyle.csv` — 1,547,896 complete records after dropping
missing values, 57 columns covering demographics, platform engagement,
health, and self-reported well-being. Source: [link or "provided for coursework"]

The dataset is not included in this repo. Place the CSV in the project root
before running.

## Approach

- **Outcome:** `daily_active_minutes_instagram`
- **Predictors:** age, perceived stress, self-reported happiness, log-transformed
  daily step count, sleep hours, weekly work hours, exercise hours, plus
  categorical demographics (gender, country, urban/rural, income, employment,
  education)
- **Feature engineering:** `log1p` transform on step count to reduce skew; a
  stress x happiness interaction term carried over from the team's R analysis
- **Preprocessing:** `StandardScaler` on numeric features, `OneHotEncoder` on
  categoricals, wrapped in a `ColumnTransformer` inside a `Pipeline` so scaling
  is fit within each CV fold rather than on the full dataset
- **Validation:** 5-fold cross-validation, shuffled, seed 42
- **Models:** Linear Regression, Ridge (alpha=1.0), Lasso (alpha=0.001),
  kNN (k=25)

Subsampled to 200,000 rows. kNN prediction cost scales with training set size,
and at 1.55M rows the full cross-validation did not complete in reasonable time.
Given the sample size, the effect on the estimates is negligible.

## Results

| Model | CV R² | CV RMSE |
|---|---|---|
| Linear Regression | 0.8756 | 38.8391 |
| Ridge | 0.8756 | 38.8391 |
| Lasso | 0.8756 | 38.8389 |
| kNN (k=25) | 0.8299 | 45.4170 |

The three linear models are indistinguishable, which indicates the predictors
are not meaningfully collinear and that regularization has nothing to shrink at
these penalty strengths. kNN underperforms, consistent with a relationship that
is close to linear in the feature space — a local-averaging method gains nothing
and loses precision to the curse of dimensionality after one-hot encoding.

The R² values are high for behavioral data, where models of self-reported
usage typically explain a far smaller share of variance. This most likely
reflects a synthetic or simulated dataset in which the outcome was generated
from the available predictors. The modeling workflow is valid regardless, but
these numbers should not be read as an estimate of how predictable real
Instagram usage is.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python rq2_model_evaluation.py
```

Writes `rq2_model_performance.csv` with the cross-validated scores.

## Attribution

Four-person capstone with Irina Maruna, Leena Hornlein, and Ashley Cox.
This repository contains my contribution: the Research Question 2 modeling
pipeline in Python. Teammates' exploratory analysis, RQ1 regression work, and
RQ3 clustering are not included here.
