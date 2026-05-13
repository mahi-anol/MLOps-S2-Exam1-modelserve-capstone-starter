"""
ModelServe — AWS Infrastructure (Pulumi)
==========================================

Provisions:
  - 1 VPC with public + private subnets
  - Internet Gateway, NAT Gateway, route tables
  - 1 Key Pair (SSH access)
  - Security Group with required ports open
  - 2 x t3.medium EC2 instances:
      * infra-host        -> Postgres, Redis, MLflow
      * monitoring-host   -> Prometheus, Grafana, FastAPI (deployed via CI/CD)
  - 1 S3 bucket (MLflow artifact registry + Grafana provisioning files)

Notes:
  - This stack does NOT create any IAM users (account has restricted access).
  - AWS credentials (configured via `aws configure`) are passed into the
    MLflow container as env vars so it can read/write the S3 bucket.
  - The Grafana dashboard JSON + dashboards.yml are uploaded to S3 by Pulumi
    so the monitoring host's user-data can pull them down at boot. This
    sidesteps the 16 KB user-data limit.

Required Pulumi config:
  - aws:region                 (e.g. ap-southeast-1)
  - publicKey                  (contents of ~/.ssh/modelserve-key.pub)
  - awsAccessKeyId   [secret]  (the same key you used in aws configure)
  - awsSecretAccessKey [secret] (the same secret you used in aws configure)
  - dockerHubUser              (Docker Hub username, used to pull API image)
  - dockerImage                (e.g. mahianol/modelserve-api:latest)
"""

import os
import pulumi
import pulumi_aws as aws

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
cfg = pulumi.Config()
public_key = cfg.require("publicKey")
aws_access_key_id = cfg.require_secret("awsAccessKeyId")
aws_secret_access_key = cfg.require_secret("awsSecretAccessKey")
docker_hub_user = cfg.get("dockerHubUser") or "mahianol"
docker_image = cfg.get("dockerImage") or f"{docker_hub_user}/modelserve-api:latest"

aws_cfg = pulumi.Config("aws")
region = aws_cfg.get("region") or "ap-southeast-1"

project = "modelserve"

# Repo root (one level up from this file's "infrastructure/" folder)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------
vpc = aws.ec2.Vpc(
    f"{project}-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={"Name": f"{project}-vpc"},
)

igw = aws.ec2.InternetGateway(
    f"{project}-igw",
    vpc_id=vpc.id,
    tags={"Name": f"{project}-igw"},
)

# Public subnet (for both EC2 instances — they need to be SSH-able and serve traffic)
public_subnet = aws.ec2.Subnet(
    f"{project}-public-subnet",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    map_public_ip_on_launch=True,
    availability_zone=f"{region}a",
    tags={"Name": f"{project}-public-subnet"},
)

# Private subnet (reserved for future workloads)
private_subnet = aws.ec2.Subnet(
    f"{project}-private-subnet",
    vpc_id=vpc.id,
    cidr_block="10.0.2.0/24",
    availability_zone=f"{region}a",
    tags={"Name": f"{project}-private-subnet"},
)

# NAT gateway (so private subnet workloads could reach the internet)
nat_eip = aws.ec2.Eip(
    f"{project}-nat-eip",
    domain="vpc",
    tags={"Name": f"{project}-nat-eip"},
)

nat_gw = aws.ec2.NatGateway(
    f"{project}-nat",
    allocation_id=nat_eip.id,
    subnet_id=public_subnet.id,
    tags={"Name": f"{project}-nat"},
    opts=pulumi.ResourceOptions(depends_on=[igw]),
)

# Public route table -> IGW
public_rt = aws.ec2.RouteTable(
    f"{project}-public-rt",
    vpc_id=vpc.id,
    routes=[{"cidr_block": "0.0.0.0/0", "gateway_id": igw.id}],
    tags={"Name": f"{project}-public-rt"},
)
aws.ec2.RouteTableAssociation(
    f"{project}-public-rt-assoc",
    subnet_id=public_subnet.id,
    route_table_id=public_rt.id,
)

# Private route table -> NAT
private_rt = aws.ec2.RouteTable(
    f"{project}-private-rt",
    vpc_id=vpc.id,
    routes=[{"cidr_block": "0.0.0.0/0", "nat_gateway_id": nat_gw.id}],
    tags={"Name": f"{project}-private-rt"},
)
aws.ec2.RouteTableAssociation(
    f"{project}-private-rt-assoc",
    subnet_id=private_subnet.id,
    route_table_id=private_rt.id,
)

# ---------------------------------------------------------------------------
# Security Group
# ---------------------------------------------------------------------------
# Open ports:
#   22   -> SSH
#   8000 -> FastAPI
#   5000 -> MLflow UI
#   3000 -> Grafana
#   9090 -> Prometheus
#   5432 -> Postgres (only within VPC ideally — but kept open here for simplicity)
#   6379 -> Redis    (only within VPC ideally — but kept open here for simplicity)
sg = aws.ec2.SecurityGroup(
    f"{project}-sg",
    vpc_id=vpc.id,
    description="ModelServe SG - SSH, API, MLflow, monitoring, Postgres, Redis",
    ingress=[
        {"protocol": "tcp", "from_port": 22,   "to_port": 22,   "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 8000, "to_port": 8000, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 5000, "to_port": 5000, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 3000, "to_port": 3000, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 9090, "to_port": 9090, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 5432, "to_port": 5432, "cidr_blocks": ["0.0.0.0/0"]},
        {"protocol": "tcp", "from_port": 6379, "to_port": 6379, "cidr_blocks": ["0.0.0.0/0"]},
    ],
    egress=[
        {"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]},
    ],
    tags={"Name": f"{project}-sg"},
)

# ---------------------------------------------------------------------------
# Key Pair
# ---------------------------------------------------------------------------
key_pair = aws.ec2.KeyPair(
    f"{project}-key",
    public_key=public_key,
    tags={"Name": f"{project}-key"},
)

# ---------------------------------------------------------------------------
# S3 Bucket (MLflow artifact registry + Grafana provisioning files)
# ---------------------------------------------------------------------------
artifact_bucket = aws.s3.BucketV2(
    f"{project}-artifacts",
    force_destroy=True,
    tags={"Name": f"{project}-artifacts"},
)

aws.s3.BucketPublicAccessBlock(
    f"{project}-artifacts-pab",
    bucket=artifact_bucket.id,
    block_public_acls=True,
    block_public_policy=True,
    ignore_public_acls=True,
    restrict_public_buckets=True,
)

# Upload Grafana provisioning files so the monitoring host can fetch them at boot.
grafana_dashboard_obj = aws.s3.BucketObject(
    f"{project}-grafana-dashboard",
    bucket=artifact_bucket.id,
    key="grafana/modelserve-dashboard.json",
    source=pulumi.FileAsset(
        os.path.join(_repo_root, "monitoring/grafana/provisioning/dashboards/modelserve-dashboard.json")
    ),
)

grafana_dashboards_yml_obj = aws.s3.BucketObject(
    f"{project}-grafana-dashboards-yml",
    bucket=artifact_bucket.id,
    key="grafana/dashboards.yml",
    source=pulumi.FileAsset(
        os.path.join(_repo_root, "monitoring/grafana/provisioning/dashboards/dashboards.yml")
    ),
)

# ---------------------------------------------------------------------------
# AMI lookup — Ubuntu 22.04 (Jammy) amd64
# ---------------------------------------------------------------------------
ubuntu_ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["099720109477"],  # Canonical
    filters=[
        {"name": "name", "values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
        {"name": "virtualization-type", "values": ["hvm"]},
        {"name": "root-device-type", "values": ["ebs"]},
        {"name": "architecture", "values": ["x86_64"]},
    ],
)


# ---------------------------------------------------------------------------
# User-data scripts
# ---------------------------------------------------------------------------
def _common_docker_install() -> str:
    """Snippet that installs Docker + Compose plugin on Ubuntu 22.04."""
    return r"""
set -eux
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker
usermod -aG docker ubuntu
"""


def make_infra_userdata(args) -> str:
    """User-data for EC2 #1: runs Postgres, Redis, MLflow."""
    bucket_name, akid, sak = args
    return f"""#!/bin/bash
{_common_docker_install()}

mkdir -p /opt/modelserve
cat > /opt/modelserve/docker-compose.yml <<'YAML'
services:
  postgres:
    image: postgres:15-alpine
    container_name: modelserve-postgres
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow_password
      POSTGRES_DB: mlflow_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: modelserve-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.19.0
    container_name: modelserve-mlflow
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      AWS_ACCESS_KEY_ID: "${{AWS_ACCESS_KEY_ID}}"
      AWS_SECRET_ACCESS_KEY: "${{AWS_SECRET_ACCESS_KEY}}"
      AWS_DEFAULT_REGION: "{region}"
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow_password@postgres:5432/mlflow_db
      MLFLOW_S3_BUCKET: "{bucket_name}"
    entrypoint: ["/bin/sh","-c"]
    command:
      - |
        pip install --no-cache-dir psycopg2-binary boto3 && \\
        mlflow server \\
          --backend-store-uri postgresql://mlflow:mlflow_password@postgres:5432/mlflow_db \\
          --default-artifact-root s3://{bucket_name}/mlflow-artifacts \\
          --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
YAML

cat > /opt/modelserve/.env <<ENV
AWS_ACCESS_KEY_ID={akid}
AWS_SECRET_ACCESS_KEY={sak}
ENV
chmod 600 /opt/modelserve/.env

cd /opt/modelserve
docker compose --env-file .env up -d
"""


def make_monitoring_userdata(args) -> str:
    """User-data for EC2 #2: runs Prometheus, Grafana, and the FastAPI image.

    Also pulls the Grafana dashboard provisioning files from S3 so the
    dashboard shows up automatically once Grafana boots.
    """
    infra_private_ip, image, bucket_name, akid, sak = args

    prom_yaml = f"""global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'modelserve-api'
    metrics_path: '/metrics'
    scrape_interval: 5s
    static_configs:
      - targets: ['api:8000']
        labels:
          service: 'modelserve-api'
"""
    grafana_ds = """apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    uid: prometheus-modelserve
    isDefault: true
    editable: true
"""
    return f"""#!/bin/bash
{_common_docker_install()}

# AWS CLI is needed to pull the dashboard JSON from S3.
apt-get install -y awscli
export AWS_ACCESS_KEY_ID="{akid}"
export AWS_SECRET_ACCESS_KEY="{sak}"
export AWS_DEFAULT_REGION="{region}"

mkdir -p /opt/modelserve/prometheus
mkdir -p /opt/modelserve/grafana/provisioning/datasources
mkdir -p /opt/modelserve/grafana/provisioning/dashboards

cat > /opt/modelserve/prometheus/prometheus.yml <<'PROM'
{prom_yaml}
PROM

cat > /opt/modelserve/grafana/provisioning/datasources/prometheus.yml <<'DS'
{grafana_ds}
DS

# Pull Grafana dashboard provisioning files from S3
aws s3 cp "s3://{bucket_name}/grafana/dashboards.yml" \\
  /opt/modelserve/grafana/provisioning/dashboards/dashboards.yml
aws s3 cp "s3://{bucket_name}/grafana/modelserve-dashboard.json" \\
  /opt/modelserve/grafana/provisioning/dashboards/modelserve-dashboard.json

# Default endpoints (private VPC IP of the infra host). CI/CD can override these
# by writing a different .env file before running `docker compose up -d api`.
cat > /opt/modelserve/.env <<ENV
MLFLOW_TRACKING_URI=http://{infra_private_ip}:5000
REDIS_HOST={infra_private_ip}
REDIS_PORT=6379
MLFLOW_MODEL_NAME=fraud-detection-model
API_IMAGE={image}
ENV

cat > /opt/modelserve/docker-compose.yml <<'YAML'
services:
  api:
    image: ${{API_IMAGE}}
    container_name: modelserve-api
    environment:
      MLFLOW_TRACKING_URI: ${{MLFLOW_TRACKING_URI}}
      MLFLOW_MODEL_NAME: ${{MLFLOW_MODEL_NAME}}
      FEAST_REPO_PATH: /app/feast_repo
      REDIS_HOST: ${{REDIS_HOST}}
      REDIS_PORT: ${{REDIS_PORT}}
    # The baked Feast config inside the image points at "redis:6379".
    # We map that hostname to the infra EC2's private IP so the API can
    # reach the real Redis without needing to rewrite the image.
    extra_hosts:
      - "redis:${{REDIS_HOST}}"
      - "mlflow:${{REDIS_HOST}}"
      - "postgres:${{REDIS_HOST}}"
    ports:
      - "8000:8000"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.53.0
    container_name: modelserve-prometheus
    ports:
      - "9090:9090"
    volumes:
      - /opt/modelserve/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.enable-lifecycle"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:11.1.0
    container_name: modelserve-grafana
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - /opt/modelserve/grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
YAML

cd /opt/modelserve
# Pull API image best-effort (it may not exist yet on first boot — CI/CD will deploy it later).
docker pull {image} || true
docker compose up -d prometheus grafana
docker compose up -d api || echo "API image not yet available — will be deployed by CI/CD."
"""


# ---------------------------------------------------------------------------
# EC2 Instances
# ---------------------------------------------------------------------------
infra_userdata = pulumi.Output.all(
    artifact_bucket.bucket, aws_access_key_id, aws_secret_access_key
).apply(make_infra_userdata)

infra_host = aws.ec2.Instance(
    f"{project}-infra-host",
    ami=ubuntu_ami.id,
    instance_type="t3.medium",
    subnet_id=public_subnet.id,
    vpc_security_group_ids=[sg.id],
    key_name=key_pair.key_name,
    associate_public_ip_address=True,
    root_block_device={"volume_size": 30, "volume_type": "gp3"},
    user_data=infra_userdata,
    tags={"Name": f"{project}-infra-host", "Role": "mlflow-redis-postgres"},
)

monitoring_userdata = pulumi.Output.all(
    infra_host.private_ip,
    pulumi.Output.from_input(docker_image),
    artifact_bucket.bucket,
    aws_access_key_id,
    aws_secret_access_key,
).apply(make_monitoring_userdata)

monitoring_host = aws.ec2.Instance(
    f"{project}-monitoring-host",
    ami=ubuntu_ami.id,
    instance_type="t3.medium",
    subnet_id=public_subnet.id,
    vpc_security_group_ids=[sg.id],
    key_name=key_pair.key_name,
    associate_public_ip_address=True,
    root_block_device={"volume_size": 30, "volume_type": "gp3"},
    user_data=monitoring_userdata,
    tags={"Name": f"{project}-monitoring-host", "Role": "prometheus-grafana-api"},
    opts=pulumi.ResourceOptions(depends_on=[
        infra_host,
        grafana_dashboard_obj,
        grafana_dashboards_yml_obj,
    ]),
)

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
pulumi.export("vpc_id", vpc.id)
pulumi.export("s3_bucket", artifact_bucket.bucket)
pulumi.export("region", region)

pulumi.export("infra_host_public_ip", infra_host.public_ip)
pulumi.export("infra_host_private_ip", infra_host.private_ip)
pulumi.export("mlflow_url", infra_host.public_ip.apply(lambda ip: f"http://{ip}:5000"))
pulumi.export("postgres_endpoint", infra_host.public_ip.apply(lambda ip: f"{ip}:5432"))
pulumi.export("redis_endpoint", infra_host.public_ip.apply(lambda ip: f"{ip}:6379"))

pulumi.export("monitoring_host_public_ip", monitoring_host.public_ip)
pulumi.export("api_url", monitoring_host.public_ip.apply(lambda ip: f"http://{ip}:8000"))
pulumi.export("grafana_url", monitoring_host.public_ip.apply(lambda ip: f"http://{ip}:3000"))
pulumi.export("prometheus_url", monitoring_host.public_ip.apply(lambda ip: f"http://{ip}:9090"))