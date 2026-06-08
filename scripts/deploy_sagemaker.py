"""
Deploy both ML models to Amazon SageMaker as hosted endpoints.

Prerequisites:
  - pip install sagemaker boto3
  - .env with SAGEMAKER_ROLE_ARN and SAGEMAKER_BUCKET
  - Trained models at ml/regression/model.joblib and ml/classification/model.joblib
  - Python 3.9 environment (matches sklearn 1.2-1 SageMaker container)

Run: python scripts/deploy_sagemaker.py

Deployment takes ~12 minutes for both endpoints.
"""
import os
import tarfile
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv, set_key
from sagemaker.sklearn.model import SKLearnModel

load_dotenv()

ROOT       = Path(__file__).parent.parent
ROLE_ARN   = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET     = os.getenv("SAGEMAKER_BUCKET")
REGION     = os.getenv("AWS_REGION", "us-east-1")
ENV_FILE   = ROOT / ".env"

SKLEARN_VERSION = "1.2-1"  # Must match local sklearn 1.2.x

if not ROLE_ARN or not BUCKET:
    raise SystemExit(
        "ERROR: Set SAGEMAKER_ROLE_ARN and SAGEMAKER_BUCKET in .env before deploying."
    )


def package_model(model_dir: Path, output_path: Path) -> Path:
    """Create a tar.gz of model.joblib for SageMaker upload."""
    with tarfile.open(output_path, "w:gz") as tf:
        tf.add(model_dir / "model.joblib", arcname="model.joblib")
    print(f"  Packaged → {output_path}")
    return output_path


def deploy(
    name: str,
    model_dir: Path,
    source_dir: Path,
    endpoint_suffix: str,
    instance_type: str = "ml.t2.medium",
) -> str:
    print(f"\n{'='*60}")
    print(f"Deploying: {name}")
    print(f"  Model dir    : {model_dir}")
    print(f"  Source dir   : {source_dir}")

    # 1. Validate prerequisites
    model_artifact = model_dir / "model.joblib"
    inference_script = source_dir / "inference.py"
    if not model_artifact.exists():
        raise FileNotFoundError(f"Train the model first: {model_artifact}")
    if not inference_script.exists():
        raise FileNotFoundError(f"Missing inference script: {inference_script}")

    # 2. Package model
    tar_path = model_dir / "model.tar.gz"
    package_model(model_dir, tar_path)

    # 3. Upload to S3
    s3 = boto3.client("s3", region_name=REGION)
    s3_key = f"sagemaker/{name}/model.tar.gz"
    print(f"  Uploading to s3://{BUCKET}/{s3_key} ...")
    s3.upload_file(str(tar_path), BUCKET, s3_key)
    model_s3_uri = f"s3://{BUCKET}/{s3_key}"

    # 4. Create SKLearnModel and deploy
    timestamp = int(time.time())
    endpoint_name = f"{endpoint_suffix}-{timestamp}"

    sklearn_model = SKLearnModel(
        model_data=model_s3_uri,
        role=ROLE_ARN,
        entry_point="inference.py",
        source_dir=str(source_dir),
        framework_version=SKLEARN_VERSION,
        py_version="py3",
        name=f"{name}-{timestamp}",
    )

    print(f"  Deploying endpoint: {endpoint_name} ({instance_type}) ...")
    print("  This takes ~5-8 minutes per endpoint...")
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    print(f"  ✅ Endpoint live: {endpoint_name}")
    return endpoint_name


# ── Deploy regression endpoint ───────────────────────────────────────────────
reg_endpoint = deploy(
    name="housing-rf",
    model_dir=ROOT / "ml" / "regression",
    source_dir=ROOT / "ml" / "regression",
    endpoint_suffix="housing-rf-endpoint",
)

# ── Deploy classification endpoint ──────────────────────────────────────────
clf_endpoint = deploy(
    name="bank-lr",
    model_dir=ROOT / "ml" / "classification",
    source_dir=ROOT / "ml" / "classification",
    endpoint_suffix="bank-lr-endpoint",
)

# ── Update .env with endpoint names ─────────────────────────────────────────
if ENV_FILE.exists():
    set_key(str(ENV_FILE), "SAGEMAKER_REGRESSION_ENDPOINT", reg_endpoint)
    set_key(str(ENV_FILE), "SAGEMAKER_CLASSIFICATION_ENDPOINT", clf_endpoint)
    print(f"\nUpdated .env with endpoint names.")

print(f"\n{'='*60}")
print("Deployment complete!")
print(f"  Regression endpoint    : {reg_endpoint}")
print(f"  Classification endpoint: {clf_endpoint}")
print("\nVerify in AWS Console: SageMaker → Inference → Endpoints")
print("Run `python scripts/delete_endpoints.py` after demo to avoid charges.")
