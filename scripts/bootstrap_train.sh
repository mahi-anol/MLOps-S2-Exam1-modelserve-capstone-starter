#!/bin/bash
# ============================================================================
# ModelServe — Script 2: Train the model locally and push to remote MLflow
# ============================================================================
# Run this on your LOCAL machine AFTER scripts/bootstrap_infra.sh has finished
# successfully.
#
# It will:
#   1. Read the public IP of the infra EC2 from Pulumi outputs
#   2. Install Python deps + DVC in a local venv
#   3. Run the DVC pipeline (ingestion -> preprocessing -> features ->
#      feast_preprocess -> training)
#      - Model gets registered into the remote MLflow registry on EC2 #1
#      - Model artifacts get uploaded to the S3 bucket
#   4. Run `feast apply` and materialize features to the remote Redis on EC2 #1
#
# Usage:
#   chmod +x scripts/bootstrap_train.sh
#   ./scripts/bootstrap_train.sh
# ============================================================================

set -euo pipefail

# Make sure pulumi is on PATH and the (empty) passphrase is set,
# matching the way bootstrap_infra.sh created the stack.
export PATH="$HOME/.pulumi/bin:$PATH"
export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-}"

echo "=== ModelServe — Script 2: Train + Register Model ==="

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- Step 1: Pulumi outputs ---
echo "[1/5] Reading Pulumi outputs for the remote MLflow + Redis..."
pushd infrastructure >/dev/null
# shellcheck disable=SC1091
source venv/bin/activate 2>/dev/null || true
pulumi login --local >/dev/null 2>&1 || true
pulumi stack select "${PULUMI_STACK:-dev}" >/dev/null

MLFLOW_URL="$(pulumi stack output mlflow_url)"
REDIS_HOST="$(pulumi stack output infra_host_public_ip)"
S3_BUCKET="$(pulumi stack output s3_bucket)"
REGION="$(pulumi stack output region)"
popd >/dev/null

echo "  MLflow:    $MLFLOW_URL"
echo "  Redis:     ${REDIS_HOST}:6379"
echo "  S3 bucket: $S3_BUCKET"

# --- Step 2: Local Python env ---
echo "[2/5] Setting up local Python environment..."
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt
pip install dvc kagglehub pyarrow boto3

# --- Step 3: Env vars for training ---
echo "[3/5] Exporting MLflow + AWS env vars..."
export MLFLOW_TRACKING_URI="$MLFLOW_URL"
export AWS_ACCESS_KEY_ID="$(aws configure get aws_access_key_id)"
export AWS_SECRET_ACCESS_KEY="$(aws configure get aws_secret_access_key)"
export AWS_DEFAULT_REGION="$REGION"

# Override the mlflow_tracking_uri in params.yaml so the trainer talks
# to the remote MLflow (not localhost). Done with a non-destructive sed:
# we restore the original value at the end.
ORIG_PARAMS="$(cat params.yaml)"
ORIG_FEAST="$(cat feast_repo/feature_store.yaml)"
trap 'printf "%s" "$ORIG_PARAMS" > params.yaml; printf "%s" "$ORIG_FEAST" > feast_repo/feature_store.yaml' EXIT
sed -i "s|mlflow_tracking_uri:.*|mlflow_tracking_uri: \"${MLFLOW_URL}\"|" params.yaml

# Also point feast at the remote Redis (the same trick — we'll restore on exit)
sed -i "s|connection_string:.*|connection_string: ${REDIS_HOST}:6379|" feast_repo/feature_store.yaml

# --- Step 4: DVC pipeline ---
echo "[4/5] Running DVC pipeline (this trains and registers the model)..."
dvc init -f --no-scm
dvc repro -f

# --- Step 5: Feast apply + materialize ---
echo "[5/5] feast apply + materialize features to remote Redis..."
pushd feast_repo >/dev/null
feast apply
popd >/dev/null
python scripts/materialize_features.py

echo ""
echo "=== Script 2 complete ==="
echo ""
echo "  - Model registered at: $MLFLOW_URL  (model: fraud-detection-model, stage: Production)"
echo "  - Features materialized to Redis at: ${REDIS_HOST}:6379"
echo "  - Model artifacts stored at: s3://${S3_BUCKET}/mlflow-artifacts"
echo ""
echo "IMPORTANT: 'feast apply' updated feast_repo/registry.db locally."
echo "Commit that file so the Docker image built by CI/CD contains the latest"
echo "feature definitions:"
echo ""
echo "    git add feast_repo/registry.db training/features.parquet training/sample_request.json"
echo "    git commit -m 'training: refresh model + feature registry'"
echo "    git push origin main"
echo ""
echo "After the push, CI/CD will build, push, and deploy the API to the"
echo "monitoring EC2 host."