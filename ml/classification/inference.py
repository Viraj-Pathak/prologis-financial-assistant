"""
SageMaker inference script for the Logistic Regression Bank Marketing model.

SageMaker calls 4 functions:
- model_fn(model_dir) -> load model from disk
- input_fn(request_body, content_type) -> parse request
- predict_fn(input_data, model) -> run prediction
- output_fn(prediction, accept) -> serialize response
"""
import json
import os
import joblib
import pandas as pd

NUMERIC_FEATURES = ["age", "balance", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def model_fn(model_dir):
    """Load the model from model_dir. Called once when endpoint starts."""
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)


def input_fn(request_body, content_type):
    """Parse the incoming request. Supports JSON."""
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
        # Fill missing columns with defaults
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "unknown"
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = 0
        return df[ALL_FEATURES]
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run prediction with probability scores."""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    return [
        {
            "subscribed": bool(pred),
            "label": "yes" if pred else "no",
            "probability": float(prob),
        }
        for pred, prob in zip(predictions, probabilities)
    ]


def output_fn(prediction, accept):
    """Serialize the prediction."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
