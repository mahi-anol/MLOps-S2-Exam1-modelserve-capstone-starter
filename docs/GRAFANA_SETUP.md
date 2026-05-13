# Grafana Dashboard Setup Guide

## Access Grafana

Open `http://localhost:3000` (or `http://<EC2-IP>:3000`).
Default credentials: `admin` / `admin`. Change the password on first login.

The Prometheus datasource is auto-provisioned — no manual datasource setup needed.

---

## Required Dashboard: ModelServe Inference Monitoring

You need at least one dashboard with panels for prediction latency, error rate, throughput, and model info.

### Step 1: Create a New Dashboard

1. Click the **+** icon in the left sidebar → **New dashboard**
2. Click **Add visualization**
3. Select **Prometheus** as the data source

### Step 2: Add Panels

Create these panels (one at a time: Add visualization → configure → Apply → repeat):

**Panel 1: Request Rate (requests/sec)**
- Title: `Prediction Throughput`
- Query: `rate(prediction_requests_total[5m])`
- Visualization: Time series
- Legend: `{{method}} {{endpoint}} {{status}}`

**Panel 2: P95 Latency**
- Title: `P95 Prediction Latency`
- Query: `histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m]))`
- Visualization: Time series
- Unit: seconds (under Standard options → Unit → Time → seconds)

**Panel 3: Average Latency**
- Title: `Average Prediction Latency`
- Query: `rate(prediction_duration_seconds_sum[5m]) / rate(prediction_duration_seconds_count[5m])`
- Visualization: Stat or Gauge
- Unit: seconds

**Panel 4: Error Rate**
- Title: `Prediction Errors`
- Query: `rate(prediction_errors_total[5m])`
- Visualization: Time series
- Legend: `{{error_type}}`

**Panel 5: Feast Hit/Miss Ratio**
- Title: `Feature Store Hit Rate`
- Query A: `rate(feast_online_store_hits_total[5m])` (legend: Hits)
- Query B: `rate(feast_online_store_misses_total[5m])` (legend: Misses)
- Visualization: Time series

**Panel 6: Model Version**
- Title: `Active Model Version`
- Query: `model_version_info`
- Visualization: Stat
- Legend: `{{version}}`

**Panel 7: Total Requests (counter)**
- Title: `Total Predictions`
- Query: `prediction_requests_total`
- Visualization: Stat
- Calculation: Last (not null)

### Step 3: Arrange Panels

Drag and resize panels into a clean layout. A common arrangement is two rows: latency metrics on top, throughput and errors on the bottom.

### Step 4: Save the Dashboard

Click the **save** icon (floppy disk) → name it `ModelServe Monitoring` → Save.

---

## Generate Load for Demo

To see data in your dashboard, generate some prediction traffic:

```bash
# Single request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @training/sample_request.json

# Load test (100 requests)
ENTITY_ID=$(python3 -c "import json; print(json.load(open('training/sample_request.json'))['entity_id'])")
for i in $(seq 1 100); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"entity_id\": $ENTITY_ID}" > /dev/null
done
echo "Sent 100 requests"
```

Wait 30 seconds, then refresh the dashboard. You should see data populating.

---

## Export Dashboard JSON (for Version Control)

After creating your dashboard:

1. Open the dashboard
2. Click the **share** icon (or Settings gear → JSON Model)
3. Copy the JSON
4. Save to `monitoring/grafana/dashboards/modelserve-dashboard.json`
5. Commit to git

This way your dashboard is reproducible even if the Grafana volume is wiped.
