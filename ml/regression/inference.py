"""
SageMaker inference script for the Random Forest housing model.

SageMaker calls 4 functions in this file:
- model_fn(model_dir) -> load model from disk
- input_fn(request_body, content_type) -> parse incoming request
- predict_fn(input_data, model) -> run prediction
- output_fn(prediction, accept) -> serialize response

Uses the sklearn framework container, which handles the HTTP plumbing.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]


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
        df = pd.DataFrame(data)[FEATURES]
        return df
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run prediction. California Housing target is in 100k USD."""
    preds = model.predict(input_data)
    return [{"predicted_value_usd": float(p) * 100000} for p in preds]


def output_fn(prediction, accept):
    """Serialize the prediction."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
