# ModelServe — Architecture Documentation

## System Overview

ModelServe is a production ML serving platform for credit card fraud detection. It uses a pre-computed feature store pattern: features are engineered offline via a DVC pipeline, materialized to Redis through Feast, and served at inference time through a FastAPI service backed by an MLflow-registered XGBoost model.

The system runs on a single EC2 instance (Option A topology) with all services containerized via Docker Compose. CI/CD is handled by GitHub Actions, infrastructure is managed by Pulumi, and observability is provided by Prometheus + Grafana.

```
┌─────────────────────────────────────────────────────────────┐
│                      EC2 Instance                           │
│                                                             │
│  ┌──────────┐   ┌────────┐   ┌─────────┐   ┌───────────┐  │
│  │ Postgres │   │  Redis │   │  MLflow │   │ Prometheus│  │
│  │ (MLflow  │   │ (Feast │   │ Tracking│   │           │  │
│  │  backend)│   │ online)│   │ Server  │   │           │  │
│  └──────────┘   └───┬────┘   └────┬────┘   └─────┬─────┘  │
│                     │             │               │         │
│              ┌──────┴─────────────┴───────┐       │         │
│              │    FastAPI Inference API    ├───────┘         │
│              │    (gunicorn + uvicorn)     │                 │
│              └────────────────────────────┘                  │
│                                                             │
│  ┌──────────┐                                               │
│  │ Grafana  │ ◄── scrapes Prometheus                        │
│  └──────────┘                                               │
└─────────────────────────────────────────────────────────────┘

Offline Pipeline (runs before serving):

  Kaggle Dataset
       │
       ▼
  data_ingestion ──► data_preprocessing ──► feature_engineering
                                                    │
                                                    ▼
                                            feast_preprocess
                                                    │
                                                    ▼
                                            model_training
                                             (logs to MLflow)
```

---

## Architecture Decision Records (ADRs)

### ADR-1: Pre-computed Feature Store Pattern

**Context:** The API needs features at inference time. Options are: (a) send raw transaction data and transform in real-time, (b) pre-compute features offline and serve from a store.

**Decision:** Use a pre-computed feature store pattern. Features are engineered during the DVC pipeline, saved to a Feast-compatible parquet, materialized to Redis, and served at inference time via the Feast SDK.

**Rationale:**
- No real-time feature engineering code duplication between training and serving
- Sub-millisecond feature retrieval from Redis
- Training/serving skew is eliminated — the exact same transformed values are used
- Feast provides a standard SDK interface, versioning, and point-in-time join semantics

**Trade-offs:**
- Features are static snapshots — new transactions don't update features until re-materialization
- Entity lookup requires a known cc_num that exists in the feature store
- Adding new features requires re-running the pipeline and re-materializing

---

### ADR-2: XGBoost with scale_pos_weight Instead of Upsampling

**Context:** The fraud detection dataset is heavily imbalanced (approx. 0.5% fraud). Options: (a) upsample the minority class, (b) use class-weighted loss, (c) SMOTE.

**Decision:** Use XGBoost's `scale_pos_weight` parameter set to 50, with `handle_imbalance: false` in the DVC pipeline.

**Rationale:**
- Upsampling inflates the dataset size, increasing training time and memory usage
- `scale_pos_weight` achieves similar effect by penalizing misclassification of the minority class
- XGBoost's native support for weighted loss is well-tested and computationally efficient
- Keeps the pipeline simpler — no synthetic data generation step

**Trade-offs:**
- Less control over the exact resampling strategy compared to SMOTE
- The weight value (50) may need tuning per dataset version

---

### ADR-3: Single EC2 Instance (Option A) Deployment

**Context:** The exam offers three deployment topologies. Option A is a single EC2 with all services co-located. Option B splits training and serving. Option C uses ECS/EKS.

**Decision:** Option A — single EC2 instance running all services via Docker Compose.

**Rationale:**
- Simplest to operate and debug during the live demo
- All services share localhost networking — no cross-instance latency
- Docker Compose provides a single `up` command to start everything
- Adequate for the exam's scale (single model, batch features, demo traffic)
- Pulumi manages the full infrastructure as code

**Trade-offs:**
- Single point of failure — if the instance dies, everything is down
- Resource contention between training and serving on the same machine
- Not suitable for production-scale traffic (would need load balancing, auto-scaling)

---

### ADR-4: MLflow Model Registry for Model Versioning

**Context:** The model needs to be versioned, loaded at API startup, and tracked across experiments. Options: (a) save model to disk as .pkl, (b) use MLflow Model Registry, (c) use a custom model store.

**Decision:** Use MLflow's Model Registry with Postgres backend and local artifact storage.

**Rationale:**
- MLflow provides experiment tracking (params, metrics, artifacts) out of the box
- Model Registry supports stage transitions (Staging → Production)
- The API loads models via `models:/{name}/Production` URI — decoupled from file paths
- UI at port 5000 enables visual comparison of experiment runs
- Required by the exam specification

**Trade-offs:**
- MLflow server adds another container to manage
- Postgres dependency for the backend store
- Model loading on startup takes a few seconds (cold start)

---

### ADR-5: Prometheus + Grafana for Observability

**Context:** The exam requires custom Prometheus metrics, alert rules, and Grafana dashboards showing prediction latency, error rates, and throughput.

**Decision:** Expose Prometheus metrics from the FastAPI app at `/metrics`, scrape with Prometheus, visualize with Grafana.

**Rationale:**
- prometheus-client library integrates directly with Python — no sidecar needed
- Counter, Histogram, and Gauge metric types cover all exam requirements
- Grafana auto-provisions the Prometheus datasource via provisioning YAML
- Alert rules are defined declaratively in `alerts.yml`
- Industry-standard stack with extensive community documentation

**Trade-offs:**
- Dashboards must be created manually in the Grafana UI (not auto-provisioned)
- Alert notification channels (email, Slack) require additional configuration
- No distributed tracing (would need Jaeger/Zipkin for that)

---

## Runbook

### Initial Setup (Fresh EC2)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/modelserve.git
cd modelserve

# 2. Run bootstrap (installs everything, runs pipeline, starts services)
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh

# 3. Verify everything is running
curl http://localhost:8000/health
curl http://localhost:5000/health
curl http://localhost:9090/-/healthy
```

### Step-by-Step Manual Setup

If bootstrap fails or you want to run steps individually:

```bash
# 1. Install Python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install dvc kagglehub pyarrow

# 2. Copy env file
cp .env.example .env

# 3. Start infrastructure
docker compose up -d postgres redis mlflow
sleep 15

# 4. Run DVC pipeline
export MLFLOW_TRACKING_URI=http://localhost:5000
dvc repro

# 5. Setup Feast
cd feast_repo && REDIS_HOST=localhost feast apply && cd ..
REDIS_HOST=localhost python scripts/materialize_features.py

# 6. Start API
docker compose up -d --build api

# 7. Start monitoring
docker compose up -d prometheus grafana
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Predict (use entity_id from sample_request.json)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @training/sample_request.json

# Predict with explanation
ENTITY_ID=$(python3 -c "import json; print(json.load(open('training/sample_request.json'))['entity_id'])")
curl "http://localhost:8000/predict/${ENTITY_ID}?explain=true"

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Running Tests

```bash
pytest app/tests/ -v
```

### Common Troubleshooting

**API returns "No features found"**
- Feast materialization may not have run. Check: `REDIS_HOST=localhost python scripts/materialize_features.py`
- The entity_id may not exist. Use one from `training/sample_request.json`

**MLflow model load fails on API startup**
- Check MLflow is healthy: `curl http://localhost:5000/health`
- Verify model exists: open http://localhost:5000, check Models tab for "fraud-detection-model" in Production stage
- Re-run training: `python training/train.py`

**Docker Compose services won't start**
- Check port conflicts: `sudo lsof -i :5000 -i :8000 -i :5432 -i :6379 -i :9090 -i :3000`
- View logs: `docker compose logs <service-name>`
- Nuclear option: `docker compose down -v && docker compose up -d`

**Feast apply fails**
- Ensure `training/features.parquet` exists (run `dvc repro` first)
- Check Redis is running: `docker compose ps redis`

### Scaling Considerations (for production, beyond exam scope)

- Replace single EC2 with ECS/EKS for horizontal scaling
- Use S3 as MLflow artifact store instead of local volume
- Add a streaming feature pipeline (Kafka → Feast) for real-time feature updates
- Put API behind an ALB with auto-scaling group
- Add Redis Sentinel or ElastiCache for HA feature store

---

## Known Limitations

1. **Features are static snapshots.** Feast serves features materialized at pipeline run time. New transactions are not reflected until the pipeline re-runs and features are re-materialized. This means predictions are based on historical feature profiles, not real-time transaction context.

2. **Single-instance deployment.** All services share one EC2. Resource contention (especially during training) can affect API latency. No redundancy or failover.

3. **No authentication on API endpoints.** The FastAPI service has no auth middleware. In production, add API key validation or OAuth2.

4. **cc_num entity lookup only.** The API can only predict for cc_num values that exist in the materialized feature store. Unknown credit cards return a 400 error.

5. **Model cold start.** The API loads the model from MLflow on startup, which takes 5-15 seconds. During this window, predictions will fail. The healthcheck endpoint reflects this state.

6. **Grafana dashboards not auto-provisioned.** Dashboards must be created manually in the Grafana UI. This is intentional — the exam requires demonstrating dashboard creation skills.

7. **Alert rules are templates.** The `alerts.yml` file contains commented-out alert rules. Students must uncomment and customize thresholds for their deployment.
