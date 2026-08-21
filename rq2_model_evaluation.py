# Data Science Capstone: Data Visualizations - Research Question 2 (Python)
# Author: John Tran
# Purpose: Model daily Instagram usage time using demographic,
#          lifestyle, and well-being predictors with
#          cross-validated regression and kNN models

# import required libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor

# load dataset
DATA_PATH = "instagram_users_lifestyle.csv"

# make sure dataset is in directory
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find {DATA_PATH}")

df = pd.read_csv(DATA_PATH)


# helper function for column names since some datasets use slightly
# different column names
# just grabs the first one that exists
def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# map outcome and main predictors
# outcome variable = daily instagram usage time
COL_Y = find_col([
    "daily_active_minutes_instagram",
    "daily_active_minutes",
    "daily_instagram_minutes",
    "instagram_minutes_per_day",
    "time_spent_on_instagram_per_day"
])

# core predictors
COL_AGE = find_col(["age", "Age"])
COL_STRESS = find_col(["perceived_stress_score", "stress_score",
                       "PerceivedStress"])
COL_HAPPY = find_col(["self_reported_happiness", "happiness_score",
                      "SelfReportedHappiness"])
COL_STEPS = find_col(["daily_steps_count", "daily_steps", "steps_per_day",
                      "step_count", "DailySteps"])

# make sure the required columns actually exist
required = {
    "outcome": COL_Y,
    "age": COL_AGE,
    "stress": COL_STRESS,
    "happiness": COL_HAPPY,
    "steps": COL_STEPS
}

missing = [k for k, v in required.items() if v is None]
if missing:
    print(f"Missing columns: {missing}")
    print(f"Available: {sorted(df.columns)}")
    raise KeyError("update column names to match your dataset")

# optional demographic and lifestyle variables
optional_cats = [
    find_col(["gender", "Gender"]),
    find_col(["country", "Country"]),
    find_col(["urban_rural", "urban_or_rural", "residence_type", "UrbanRural"]),
    find_col(["income_level", "IncomeLevel"]),
    find_col(["employment_status", "EmploymentStatus"]),
    find_col(["education_level", "EducationLevel"])
]

optional_nums = [
    find_col(["sleep_hours_per_night", "sleep_hours", "SleepHours"]),
    find_col(["weekly_work_hours", "work_hours", "WorkHours"]),
    find_col(["exercise_hours_per_week", "exercise_hours", "ExerciseHours"])
]

# remove anything that didn't exist in the dataset
optional_cats = [c for c in optional_cats if c is not None]
optional_nums = [c for c in optional_nums if c is not None]

# feature engineering
# copy the data frame so we don't overwrite anything by accident
df = df.copy()

# log transform steps to reduce skew
df["log_steps"] = np.log1p(df[COL_STEPS])

# interaction term used in the final rq2 model
df["stress_x_happiness"] = df[COL_STRESS] * df[COL_HAPPY]

# set up x (predictors) and y (outcome)
y = df[COL_Y].astype(float)

numeric_feats = [
    COL_AGE,
    COL_STRESS,
    COL_HAPPY,
    "log_steps",
    "stress_x_happiness"
] + optional_nums

cat_feats = optional_cats

X = df[numeric_feats + cat_feats].copy()

model_df = pd.concat([X, y.rename("Y")], axis=1).dropna()

# subsample for tractable kNN runtime at this dataset size
model_df = model_df.sample(n=200000, random_state=42)

X = model_df.drop(columns=["Y"])
y = model_df["Y"]

print(f"Using {len(model_df)} samples")

# preprocessing
# scale numeric features
# one-hot encode categorical ones
preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ],
    remainder="drop"
)

# define models used in rq2
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=5000),
    "kNN": KNeighborsRegressor(n_neighbors=25),
}

# cross-validation setup
# 5-fold cv
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# evaluate using r2 and rmse
scoring = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error"
}

# run models and collect results
results = []
for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    mean_r2 = float(np.mean(scores["test_r2"]))
    mean_rmse = float(-np.mean(scores["test_rmse"]))
    results.append((name, mean_r2, mean_rmse))

# print results
print("\nRQ2 Model Performance (5-fold CV)")
print("Model".ljust(18) + "R²".rjust(10) + "RMSE".rjust(12))
for name, r2, rmse in results:
    print(name.ljust(18) + f"{r2:10.4f}" + f"{rmse:12.4f}")

# save results so they're easy to reference later
out = pd.DataFrame(results, columns=["model", "cv_r2", "cv_rmse"])
out.to_csv("rq2_model_performance.csv", index=False)