#!/bin/bash
# ============================================================================
# ModelServe — Script 1: Provision AWS Infrastructure (Pulumi)
# ============================================================================
# Run this on your LOCAL machine.
#
# It will:
#   1. Install AWS CLI v2 (if missing)
#   2. Install Pulumi (if missing)
#   3. Run `aws configure` so you can enter your credentials
#   4. Generate an SSH key for the EC2 hosts (if missing)
#   5. Run `pulumi up` to provision:
#        - 1 VPC, public + private subnets, IGW, NAT, route tables
#        - Security group, key pair
#        - 1 S3 bucket (MLflow artifact registry)
#        - EC2 #1 (t3.medium) running Postgres + Redis + MLflow
#        - EC2 #2 (t3.medium) running Prometheus + Grafana + FastAPI
#   6. Print the connection URLs
#
# IMPORTANT: This script does NOT create any IAM users or roles. The same
# AWS credentials you provide here are passed into the MLflow container
# (as env vars) so it can read/write the S3 bucket.
#
# Usage:
#   chmod +x scripts/bootstrap_infra.sh
#   ./scripts/bootstrap_infra.sh
# ============================================================================

set -euo pipefail

REGION="${AWS_REGION:-ap-southeast-1}"
STACK="${PULUMI_STACK:-dev}"
KEY_PATH="${HOME}/.ssh/modelserve-key"
DOCKER_HUB_USER="${DOCKER_HUB_USER:-mahianol}"
DOCKER_IMAGE="${DOCKER_IMAGE:-${DOCKER_HUB_USER}/modelserve-api:latest}"

echo "=== ModelServe — Script 1: AWS Infrastructure ==="

# --- Step 1: AWS CLI ---
if ! command -v aws >/dev/null 2>&1; then
  echo "[1/5] Installing AWS CLI v2..."
  TMPDIR="$(mktemp -d)"
  pushd "$TMPDIR" >/dev/null
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
  unzip -q awscliv2.zip
  sudo ./aws/install || sudo ./aws/install --update
  popd >/dev/null
  rm -rf "$TMPDIR"
else
  echo "[1/5] AWS CLI already installed: $(aws --version)"
fi

# --- Step 2: Pulumi ---
if ! command -v pulumi >/dev/null 2>&1; then
  echo "[2/5] Installing Pulumi..."
  curl -fsSL https://get.pulumi.com | sh
  export PATH="$HOME/.pulumi/bin:$PATH"
else
  echo "[2/5] Pulumi already installed: $(pulumi version)"
fi

# --- Step 3: AWS credentials ---
echo "[3/5] Configuring AWS credentials..."
echo "Enter your AWS credentials when prompted (used by Pulumi AND MLflow):"
aws configure

# Pull the configured credentials back out so we can pass them into Pulumi config
AWS_ACCESS_KEY_ID="$(aws configure get aws_access_key_id)"
AWS_SECRET_ACCESS_KEY="$(aws configure get aws_secret_access_key)"
REGION="$(aws configure get region || echo $REGION)"

if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
  echo "ERROR: aws configure did not yield credentials. Aborting."
  exit 1
fi

# Quick sanity check (does NOT create resources — just verifies the credentials work)
aws sts get-caller-identity >/dev/null || {
  echo "ERROR: AWS credentials are invalid. Run 'aws configure' again."
  exit 1
}

# --- Step 4: SSH key ---
if [[ ! -f "$KEY_PATH" ]]; then
  echo "[4/5] Generating SSH key at $KEY_PATH ..."
  ssh-keygen -t rsa -b 4096 -N "" -f "$KEY_PATH"
else
  echo "[4/5] SSH key already exists at $KEY_PATH"
fi
chmod 600 "$KEY_PATH"

# --- Step 5: Pulumi up ---
echo "[5/5] Running pulumi up..."
cd "$(dirname "$0")/../infrastructure"

# Local Python venv for the Pulumi program
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# Use local file backend (no Pulumi Cloud / no extra account needed)
pulumi login --local

# Create the stack if it doesn't exist (passphrase set to empty for unattended use)
export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-}"
pulumi stack init "$STACK" 2>/dev/null || pulumi stack select "$STACK"

# Configure the stack
pulumi config set aws:region "$REGION"
pulumi config set publicKey "$(cat ${KEY_PATH}.pub)"
pulumi config set --secret awsAccessKeyId "$AWS_ACCESS_KEY_ID"
pulumi config set --secret awsSecretAccessKey "$AWS_SECRET_ACCESS_KEY"
pulumi config set dockerHubUser "$DOCKER_HUB_USER"
pulumi config set dockerImage   "$DOCKER_IMAGE"

pulumi up --yes

echo ""
echo "=== Script 1 complete ==="
echo ""
pulumi stack output
echo ""
echo "Infrastructure is up. EC2 user-data is still starting the containers — "
echo "give it ~2-3 minutes for MLflow / Postgres / Redis / Grafana to be ready."
echo ""
echo "Next: run scripts/bootstrap_train.sh to train the model against the remote MLflow."
