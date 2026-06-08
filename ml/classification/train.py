"""
Train a Logistic Regression classifier on the UCI Bank Marketing dataset.

Steps:
  1. Download / load the dataset (cached locally after first run)
  2. EDA: shape, class balance, feature types
  3. ColumnTransformer: StandardScaler (numeric) + OneHotEncoder (categorical)
  4. 80/20 train-test split (stratified)
  5. Fit LogisticRegression(class_weight="balanced", max_iter=1000)
  6. Evaluate: accuracy, precision, recall, F1, confusion matrix
  7. Save model to ml/classification/model.joblib
  8. Save metrics to ml/classification/metrics.json

Run: python ml/classification/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT_DIR = Path(__file__).parent
CACHE_PATH = OUT_DIR / "bank_marketing.csv"

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
)

# - 1. Load / download dataset -
def load_data() -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"Loading cached dataset from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH, sep=";")

    print("Downloading UCI Bank Marketing dataset...")
    import io, zipfile
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(name), sep=";")
    df.to_csv(CACHE_PATH, index=False, sep=";")
    print(f"Saved to {CACHE_PATH}")
    return df


df = load_data()
print(f"\nDataset shape: {df.shape}")
print(f"Target distribution:\n{df['y'].value_counts(normalize=True).round(3)}")
print(f"\n- EDA -")
print(df.describe(include="all").to_string())

# - 2. Feature / target split -
X = df.drop(columns=["y"])
y = (df["y"] == "yes").astype(int)

numeric_features  = ["age", "balance", "duration", "campaign", "pdays", "previous"]
categorical_features = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]

print(f"\nNumeric features  : {numeric_features}")
print(f"Categorical features: {categorical_features}")
print(f"Class balance - no: {(y==0).mean():.1%}  yes: {(y==1).mean():.1%}")

# - 3. Train / test split -
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# - 4. Pipeline -
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )),
])

print("\nTraining Logistic Regression classifier...")
pipeline.fit(X_train, y_train)

# - 5. Evaluation -
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc  = float(accuracy_score(y_test, y_pred))
prec = float(precision_score(y_test, y_pred, zero_division=0))
rec  = float(recall_score(y_test, y_pred))
f1   = float(f1_score(y_test, y_pred))
cm   = confusion_matrix(y_test, y_pred).tolist()

print(f"\n- Test Set Metrics -")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"\nConfusion Matrix:\n{np.array(cm)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['no','yes'])}")

# - 6. Save model artifact -
model_path = OUT_DIR / "model.joblib"
joblib.dump(pipeline, model_path)
print(f"\nModel saved - {model_path}")

# - 7. Save metrics JSON -
metrics = {
    "model_type": "LogisticRegression",
    "dataset": "UCI Bank Marketing",
    "n_train": len(X_train),
    "n_test": len(X_test),
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "confusion_matrix": cm,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
}
metrics_path = OUT_DIR / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"Metrics saved - {metrics_path}")

# - 8. Sanity-check prediction -
sample = X_test.iloc[[0]]
pred_label = pipeline.predict(sample)[0]
pred_prob  = pipeline.predict_proba(sample)[0, 1]
actual     = y_test.iloc[0]
print(f"\nSanity check:")
print(f"  Predicted : {'yes' if pred_label else 'no'}  (prob={pred_prob:.3f})")
print(f"  Actual    : {'yes' if actual else 'no'}")
