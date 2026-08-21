"""
Corrected end-to-end training pipeline for the Advertising Sales project.
Fixes found in the original notebooks:
  - Y was a DataFrame (df[['Sales']]) instead of a Series (df['Sales'])
  - No RandomizedSearchCV / cross-validation was present anywhere
  - No model serialization step was present anywhere
Run from the project root (expects advertising_cleaned.csv alongside this file,
or adjust DATA_PATH below).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path(__file__).parent / "advertising_cleaned.csv"
MODEL_OUT = Path(__file__).parent / "best_random_forest_advertising.joblib"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"

X = df[FEATURES]
y = df[TARGET]  # Series, not df[[TARGET]] -- avoids downstream shape warnings

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 2. Baseline: Linear Regression
# ---------------------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("=== Linear Regression (baseline) ===")
print(f"R2:   {r2_score(y_test, lr_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, lr_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.4f}")
print(f"Coefficients: {dict(zip(FEATURES, lr.coef_))}")
print(f"Intercept: {lr.intercept_:.4f}\n")

# ---------------------------------------------------------------------------
# 3. Random Forest + hyperparameter search
# ---------------------------------------------------------------------------
param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 8, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, "sqrt", "log2"],
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=40,
    scoring="r2",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
search.fit(X_train, y_train)

best_rf = search.best_estimator_
rf_pred = best_rf.predict(X_test)

print("=== Random Forest (tuned) ===")
print(f"Best params: {search.best_params_}")
print(f"R2:   {r2_score(y_test, rf_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, rf_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.4f}")
print(f"Feature importances: {dict(zip(FEATURES, best_rf.feature_importances_))}\n")

# ---------------------------------------------------------------------------
# 4. Cross-validate the tuned model on the FULL dataset (report robustness)
# ---------------------------------------------------------------------------
cv_scores = cross_val_score(best_rf, X, y, cv=cv, scoring="r2")
print("=== 5-fold CV (tuned RF, full dataset) ===")
print(f"Fold scores: {np.round(cv_scores, 4)}")
print(f"Mean R2: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})\n")

# ---------------------------------------------------------------------------
# 5. Serialize the winning model
# ---------------------------------------------------------------------------
joblib.dump(best_rf, MODEL_OUT)
print(f"Saved best model to: {MODEL_OUT}")
