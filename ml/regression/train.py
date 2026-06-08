"""
Train a Random Forest Regressor on the California Housing dataset.

Steps:
  1. Load dataset from scikit-learn
  2. Exploratory data analysis (shape, stats, target distribution)
  3. StandardScaler preprocessing inside a Pipeline
  4. 80/20 train-test split
  5. Fit RandomForestRegressor(n_estimators=100, max_depth=15)
  6. Evaluate: RMSE, MAE, R-
  7. Save model to ml/regression/model.joblib
  8. Save metrics to ml/regression/metrics.json

Run: python ml/regression/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).parent

# - 1. Load dataset -
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target

print(f"  Shape: {X.shape}  |  Target: median house value (100k USD)")
print(f"  Features: {list(X.columns)}")
print("\n=== EDA ===")
print(pd.concat([X, y.rename("MedHouseVal")], axis=1).describe().round(3).to_string())

# - 2. Train / test split -
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# - 3. Pipeline: StandardScaler - RandomForestRegressor -
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )),
])

print("\nTraining Random Forest Regressor...")
pipeline.fit(X_train, y_train)

# - 4. Evaluation -
y_pred = pipeline.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))
r2   = float(r2_score(y_test, y_pred))

print("\n=== Test Set Metrics ===")
print(f"  RMSE : {rmse:.4f} (100k USD) = ${rmse * 100000:,.0f}")
print(f"  MAE  : {mae:.4f}  (100k USD) = ${mae  * 100000:,.0f}")
print(f"  R2   : {r2:.4f}")

# - 5. Feature importance -
rf_model = pipeline.named_steps["model"]
importances = dict(
    sorted(
        zip(housing.feature_names, rf_model.feature_importances_.tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )
)
print("\nFeature importances:")
for feat, imp in importances.items():
    print(f"  {feat:<15} {imp:.4f}")

# - 6. Save model artifact -
model_path = OUT_DIR / "model.joblib"
joblib.dump(pipeline, model_path)
print(f"\nModel saved - {model_path}")

# - 7. Save metrics JSON -
metrics = {
    "model_type": "RandomForestRegressor",
    "dataset": "California Housing (sklearn)",
    "n_train": len(X_train),
    "n_test": len(X_test),
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "feature_names": housing.feature_names,
    "feature_importance": importances,
}
metrics_path = OUT_DIR / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"Metrics saved - {metrics_path}")

# - 8. Sanity-check prediction -
sample = X_test.iloc[[0]]
pred_val = pipeline.predict(sample)[0]
actual_val = y_test.iloc[0]
print(f"\nSanity check:")
print(f"  Predicted : ${pred_val * 100000:,.0f}")
print(f"  Actual    : ${actual_val * 100000:,.0f}")
