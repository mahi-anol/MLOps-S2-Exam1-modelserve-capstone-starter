# Grafana Dashboard Setup Guide

## Access Grafana

Grafana runs on the **monitoring-host** EC2 (EC2 #2). Open:

```
http://<monitoring_host_public_ip>:3000
```

(or `http://localhost:3000` for the local-compose flow.) Default credentials: `admin` / `admin`. Change the password on first login.

Both the Prometheus datasource **and** the ModelServe dashboard are auto-provisioned from `monitoring/grafana/provisioning/` — when Grafana starts you'll find a dashboard called **ModelServe Monitoring** already loaded, pre-wired to the Prometheus datasource. No manual setup needed for the default panels.

The rest of this guide covers (1) what's in the provisioned dashboard, (2) how to add custom panels on top, and (3) how to keep your changes in version control.

---

## What's in the provisioned dashboard

`monitoring/grafana/provisioning/dashboards/modelserve-dashboard.json` ships with panels covering all metrics the FastAPI service exposes:

| Panel                          | PromQL                                                                                       |
|--------------------------------|----------------------------------------------------------------------------------------------|
| Prediction Throughput          | `rate(prediction_requests_total[5m])`                                                        |
| P95 Prediction Latency         | `histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m]))`                     |
| Average Prediction Latency     | `rate(prediction_duration_seconds_sum[5m]) / rate(prediction_duration_seconds_count[5m])`    |
| Prediction Errors              | `rate(prediction_errors_total[5m])`  (legend: `{{error_type}}`)                              |
| Feature Store Hit Rate         | `rate(feast_online_store_hits_total[5m])` vs `rate(feast_online_store_misses_total[5m])`     |
| Active Model Version           | `model_version_info`  (legend: `{{version}}`)                                                |
| Total Predictions              | `prediction_requests_total`                                                                  |

The `error_type` label on `prediction_errors_total` includes `feature_not_found`, `prediction_error`, and `rollback_error`, so the errors panel distinguishes feature-store misses, model-inference failures, and failed rollback attempts.

The **Active Model Version** panel is wired to the same `model_version_info` gauge that the `POST /rollback` endpoint updates — so when you roll back to a previous version, the panel re-paints with the new version label within one scrape interval (≤5s by default).

---

## Adding custom panels

If you want to extend the dashboard:

1. Open the **ModelServe Monitoring** dashboard.
2. Click **Add → Visualization**.
3. Select **Prometheus** as the data source (already provisioned).
4. Add a query, configure the panel, click **Apply**.

A few useful extras that aren't in the default set:

**Rollback events over time**
- Query: `changes(model_version_info[1h])`
- Visualization: Time series
- Counts how often the active model version label changes — useful for spotting frequent rollbacks.

**Failed rollback attempts**
- Query: `rate(prediction_errors_total{error_type="rollback_error"}[5m])`
- Visualization: Stat or Time series
- Flags rollbacks that hit a bad version or registry error.

**Endpoint-level error rate**
- Query: `sum by (endpoint) (rate(prediction_requests_total{status="error"}[5m]))`
- Visualization: Time series
- Splits errors by `/predict` vs `/predict/{entity_id}`.

---

## Generate load for a demo

To populate the panels with data, generate some prediction traffic against the API:

```bash
API=http://<monitoring_host_public_ip>:8000   # or http://localhost:8000 locally

# Single request
curl -X POST $API/predict \
  -H "Content-Type: application/json" \
  -d @training/sample_request.json

# Load test (100 requests)
ENTITY_ID=$(python3 -c "import json; print(json.load(open('training/sample_request.json'))['entity_id'])")
for i in $(seq 1 100); do
  curl -s -X POST $API/predict \
    -H "Content-Type: application/json" \
    -d "{\"entity_id\": $ENTITY_ID}" > /dev/null
done
echo "Sent 100 requests"
```

Wait one scrape interval (Prometheus is configured for 5s on the `modelserve-api` job), then refresh the dashboard.

To demo the **Active Model Version** panel updating, trigger a rollback while a load loop is running:

```bash
curl -X POST $API/rollback
```

The panel re-paints with the new version label within ~5 seconds, and `Prediction Throughput` keeps climbing — proving the swap was in-process and didn't drop traffic.

---

## Persisting UI changes back to version control

The dashboard JSON in the repo is what Grafana loads on startup. If you tweak panels in the UI and want those changes to survive a container rebuild:

1. Open the dashboard.
2. Click the **share** icon (or **Dashboard settings → JSON Model**).
3. Copy the JSON.
4. Overwrite `monitoring/grafana/provisioning/dashboards/modelserve-dashboard.json` with it.
5. Commit and push — the next CI deploy (or a `docker compose up -d --force-recreate grafana`) will pick up the new dashboard.

This keeps the dashboard reproducible across Grafana volume wipes and EC2 replacements.

---

## Alerts

Prometheus alert rules live in `monitoring/prometheus/alerts.yml` and are loaded automatically by the Prometheus container. The shipped rules cover:

- **HighLatencyP95** — P95 latency > 500ms for 1 minute (warning)
- **HighErrorRate** — `prediction_errors_total` rate > 0.1/s for 2 minutes (critical)
- **ServiceDown** — `up{job="modelserve-api"} == 0` for 1 minute (critical)

You can view firing alerts at `http://<monitoring_host_public_ip>:9090/alerts`. Routing them to Slack / email requires running Alertmanager — not configured in this repo.