# API Quick Reference

## 🚀 8 Production Endpoints

### 1. Single Vehicle Prediction
```http
POST /predict/soot-load
```
**Use:** Predict soot load for one vehicle  
**Input:** Recent sensor readings (1-100 readings)  
**Output:** Soot %, risk level, regeneration recommendation

### 2. Batch Prediction
```http
POST /predict/batch
```
**Use:** Predict for multiple vehicles simultaneously  
**Input:** Array of vehicle reading sets  
**Output:** Aggregated results with processing time

### 3. Model Info
```http
GET /model/info
```
**Use:** Get model metadata  
**Output:** Version, metrics (recall: 96.2%, precision: 95.9%), threshold

### 4. Model Features
```http
GET /model/features
```
**Use:** Get feature importance rankings  
**Output:** 44 features with importance scores, top 5 highlighted

### 5. Fleet Status
```http
GET /fleet/status
```
**Use:** Get fleet-wide risk summary  
**Output:** Vehicle counts by risk level, last update timestamp

### 6. Vehicle History
```http
GET /vehicles/{vehicle_id}/history?days=7
```
**Use:** Get prediction history for specific vehicle  
**Params:** `days` (1-90)  
**Output:** Historical predictions over time period

### 7. Maintenance Alert
```http
POST /vehicles/{vehicle_id}/alert?alert_type=regeneration&priority=4
```
**Use:** Create maintenance alert for vehicle  
**Params:** `alert_type` (regeneration|inspection|urgent), `priority` (1-5)  
**Output:** Alert confirmation with timestamp

### 8. Health Check
```http
GET /health
```
**Use:** Check API health status  
**Output:** Status, model_loaded, cache_available, uptime

### 9. Prometheus Metrics
```http
GET /metrics
```
**Use:** Scrape metrics for monitoring  
**Output:** Prometheus text format (predictions, latency, cache hits)

### 10. API Root
```http
GET /
```
**Use:** List all available endpoints  
**Output:** Endpoint directory with links

---

## 📊 Response Examples

### Single Prediction Response
```json
{
  "vehicle_id": "VEH-001",
  "soot_load_percent": 23.45,
  "risk_level": "LOW",
  "confidence": 0.982,
  "regeneration_recommended": false,
  "estimated_km_to_failure": 875.2,
  "prediction_timestamp": "2026-01-18T10:30:15",
  "features_used": 44
}
```

### Model Info Response
```json
{
  "model_version": "v1.0.0",
  "training_date": "2026-01-18",
  "model_type": "XGBoost",
  "num_features": 44,
  "optimal_threshold": 0.696,
  "performance_metrics": {
    "recall": 0.962,
    "precision": 0.959,
    "f1_score": 0.961,
    "roc_auc": 0.9993
  }
}
```

### Feature Importance Response
```json
{
  "total_features": 44,
  "top_5": [
    {
      "name": "Trip distance (km)",
      "importance": 0.145,
      "rank": 1
    },
    {
      "name": "Exhaust temperature - DPF inlet_std",
      "importance": 0.112,
      "rank": 2
    },
    ...
  ]
}
```

---

## 🎯 Common Use Cases

### Maintenance Scheduler
```python
# Check fleet status
response = requests.get(f"{BASE_URL}/fleet/status")
high_risk_count = response.json()["high_risk_count"]

# Get predictions for high-risk vehicles
for vehicle_id in high_risk_vehicles:
    pred = requests.post(f"{BASE_URL}/predict/soot-load", json={...})
    if pred.json()["regeneration_recommended"]:
        # Create alert
        requests.post(
            f"{BASE_URL}/vehicles/{vehicle_id}/alert",
            params={"alert_type": "regeneration", "priority": 4}
        )
```

### Real-time Monitoring Dashboard
```python
# Poll metrics every 15 seconds
while True:
    metrics = requests.get(f"{BASE_URL}/metrics").text
    
    # Parse Prometheus metrics
    predictions_rate = parse_metric(metrics, "predictions_total")
    p95_latency = parse_metric(metrics, "prediction_latency_seconds", "0.95")
    cache_hit_rate = calculate_cache_rate(metrics)
    
    dashboard.update(predictions_rate, p95_latency, cache_hit_rate)
    time.sleep(15)
```

### Historical Analysis
```python
# Analyze vehicle trends
history = requests.get(f"{BASE_URL}/vehicles/VEH-001/history?days=30").json()

# Plot soot load over time
timestamps = [pred["timestamp"] for pred in history["predictions"]]
soot_loads = [pred["soot_load_percent"] for pred in history["predictions"]]

plt.plot(timestamps, soot_loads)
plt.title("VEH-001 Soot Accumulation (30 days)")
plt.show()
```

---

## 🔧 cURL Examples

### Make Single Prediction
```bash
curl -X POST http://localhost:8000/predict/soot-load \
  -H "Content-Type: application/json" \
  -d @sensor_data.json
```

### Get Top Features
```bash
curl http://localhost:8000/model/features | jq '.top_5'
```

### Check Health
```bash
curl http://localhost:8000/health | jq .
```

### Create Alert
```bash
curl -X POST "http://localhost:8000/vehicles/VEH-001/alert?alert_type=urgent&priority=5"
```

---

## 📝 Interactive Documentation

Access Swagger UI for interactive testing:
```
http://localhost:8000/docs
```

Features:
- ✅ Try all endpoints in browser
- ✅ Auto-generated request examples
- ✅ Response schema validation
- ✅ Authentication testing (when enabled)
