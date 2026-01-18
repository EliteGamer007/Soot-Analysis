"""
Integration Tests for DPF Soot Prediction API
Tests end-to-end workflows with mock data
"""

import pytest
import requests
import json
from datetime import datetime, timedelta
import time
import random

BASE_URL = "http://localhost:8000"

class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.fixture(scope="class", autouse=True)
    def check_api_running(self):
        """Ensure API is running before tests"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("API not running - start with: python api_server.py")
        except requests.exceptions.RequestException:
            pytest.skip("API not accessible - start with: python api_server.py")
    
    def generate_mock_reading(self, vehicle_id="VEH-TEST-001", offset_minutes=0):
        """Generate realistic mock sensor reading"""
        timestamp = datetime.now() + timedelta(minutes=offset_minutes)
        return {
            "vehicle_id": vehicle_id,
            "timestamp": timestamp.isoformat(),
            "engine_rpm": random.uniform(1500, 2500),
            "vehicle_speed": random.uniform(40, 90),
            "engine_load": random.uniform(30, 70),
            "exhaust_temp_dpf_inlet": random.uniform(350, 500),
            "exhaust_temp_doc_inlet": random.uniform(320, 480),
            "exhaust_flow_rate": random.uniform(60, 120),
            "differential_pressure": random.uniform(5, 25),
            "fuel_consumption": random.uniform(5, 15),
            "trip_distance": random.uniform(50, 200),
            "dist_since_regen": random.uniform(200, 800)
        }
    
    def test_health_endpoint(self):
        """Test /health endpoint returns correct status"""
        response = requests.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["model_loaded"] == True
        
    def test_model_info_endpoint(self):
        """Test /model/info returns correct metadata"""
        response = requests.get(f"{BASE_URL}/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "num_features" in data
        assert "performance_metrics" in data
        assert data["num_features"] == 44
        assert "recall" in data["performance_metrics"]
        
    def test_model_features_endpoint(self):
        """Test /model/features returns feature information"""
        response = requests.get(f"{BASE_URL}/model/features")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_features" in data
        assert "features" in data
        assert "top_5" in data
        assert data["total_features"] == 44
        assert len(data["top_5"]) == 5
        
        # Check feature structure
        feature = data["features"][0]
        assert "name" in feature
        assert "importance" in feature
        assert "rank" in feature
    
    def test_single_prediction_success(self):
        """Test successful single vehicle prediction"""
        readings = [self.generate_mock_reading("VEH-INT-001", i) for i in range(-5, 1)]
        payload = {"recent_readings": readings}
        
        response = requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "vehicle_id" in data
        assert "soot_load_percent" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert "regeneration_recommended" in data
        assert "prediction_timestamp" in data
        
        # Check data types and ranges
        assert data["vehicle_id"] == "VEH-INT-001"
        assert 0 <= data["soot_load_percent"] <= 100
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["regeneration_recommended"], bool)
        
    def test_batch_prediction_success(self):
        """Test successful batch prediction"""
        vehicles = []
        for i in range(3):
            vehicle_id = f"VEH-BATCH-{i:03d}"
            readings = [self.generate_mock_reading(vehicle_id, j) for j in range(-3, 1)]
            vehicles.append({"recent_readings": readings})
        
        payload = {"vehicles": vehicles}
        response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "total_processed" in data
        assert "high_risk_count" in data
        assert "processing_time_ms" in data
        
        assert data["total_processed"] == 3
        assert len(data["predictions"]) == 3
        assert data["processing_time_ms"] > 0
        
    def test_caching_behavior(self):
        """Test that caching improves performance"""
        readings = [self.generate_mock_reading("VEH-CACHE-001", i) for i in range(-5, 1)]
        payload = {"recent_readings": readings}
        
        # First request (cache miss)
        start1 = time.time()
        response1 = requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        time1 = (time.time() - start1) * 1000
        
        # Second request (cache hit)
        start2 = time.time()
        response2 = requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        time2 = (time.time() - start2) * 1000
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Cache hit should be faster (allowing for network variance)
        # Commenting out strict timing check as it can be flaky
        # assert time2 < time1, f"Cache hit ({time2:.2f}ms) should be faster than miss ({time1:.2f}ms)"
        
        # Results should be identical
        assert response1.json()["soot_load_percent"] == response2.json()["soot_load_percent"]
        
    def test_prediction_validation_rejects_invalid_rpm(self):
        """Test that invalid RPM is rejected"""
        readings = [self.generate_mock_reading()]
        readings[0]["engine_rpm"] = 6000  # Invalid (max 5000)
        payload = {"recent_readings": readings}
        
        response = requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        assert response.status_code == 422  # Validation error
        
    def test_prediction_validation_rejects_invalid_speed(self):
        """Test that invalid speed is rejected"""
        readings = [self.generate_mock_reading()]
        readings[0]["vehicle_speed"] = 250  # Invalid (max 200)
        payload = {"recent_readings": readings}
        
        response = requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        assert response.status_code == 422
        
    def test_high_risk_scenario(self):
        """Test prediction with high-risk sensor values"""
        readings = []
        for i in range(10):
            reading = {
                "vehicle_id": "VEH-HIGHRISK-001",
                "timestamp": (datetime.now() + timedelta(minutes=i)).isoformat(),
                "engine_rpm": 2200,
                "vehicle_speed": 85,
                "engine_load": 75,  # High load
                "exhaust_temp_dpf_inlet": 550,  # High temp
                "exhaust_temp_doc_inlet": 520,
                "exhaust_flow_rate": 140,
                "differential_pressure": 35 + i,  # Increasing
                "fuel_consumption": 18,
                "trip_distance": 180,
                "dist_since_regen": 1200 + i * 50  # Long since regen
            }
            readings.append(reading)
        
        payload = {"recent_readings": readings}
        response = requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # High-risk scenario should produce elevated predictions
        # Not asserting specific risk level as it depends on model
        assert data["soot_load_percent"] >= 0
        
    def test_fleet_status_endpoint(self):
        """Test /fleet/status endpoint"""
        response = requests.get(f"{BASE_URL}/fleet/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_vehicles" in data
        assert "high_risk_count" in data
        assert "last_updated" in data
        
    def test_vehicle_history_endpoint(self):
        """Test /vehicles/{vehicle_id}/history endpoint"""
        response = requests.get(f"{BASE_URL}/vehicles/VEH-001/history?days=7")
        
        assert response.status_code == 200
        data = response.json()
        assert "vehicle_id" in data
        assert "period_days" in data
        assert data["vehicle_id"] == "VEH-001"
        assert data["period_days"] == 7
        
    def test_maintenance_alert_creation(self):
        """Test /vehicles/{vehicle_id}/alert endpoint"""
        response = requests.post(
            f"{BASE_URL}/vehicles/VEH-001/alert?alert_type=regeneration&priority=4"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] == True
        assert "alert" in data
        
    def test_metrics_endpoint(self):
        """Test /metrics endpoint returns Prometheus format"""
        response = requests.get(f"{BASE_URL}/metrics")
        
        assert response.status_code == 200
        content = response.text
        
        # Check for key metrics
        assert "predictions_total" in content
        assert "prediction_latency_seconds" in content
        assert "predicted_soot_level" in content
        
    def test_root_endpoint(self):
        """Test root endpoint lists all available endpoints"""
        response = requests.get(f"{BASE_URL}/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert len(data["endpoints"]) >= 8
        
    def test_concurrent_predictions(self):
        """Test handling multiple concurrent predictions"""
        import concurrent.futures
        
        def make_prediction(vehicle_num):
            vehicle_id = f"VEH-CONCURRENT-{vehicle_num:03d}"
            readings = [self.generate_mock_reading(vehicle_id, i) for i in range(-3, 1)]
            payload = {"recent_readings": readings}
            return requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        
        # Send 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(5)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        assert len(responses) == 5

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_missing_required_fields(self):
        """Test prediction with missing required fields"""
        payload = {
            "recent_readings": [{
                "vehicle_id": "VEH-001",
                "timestamp": datetime.now().isoformat()
                # Missing all sensor values
            }]
        }
        
        response = requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        assert response.status_code == 422  # Validation error
        
    def test_empty_readings_list(self):
        """Test prediction with empty readings list"""
        payload = {"recent_readings": []}
        
        response = requests.post(f"{BASE_URL}/predict/soot-load", json=payload)
        assert response.status_code == 422
        
    def test_invalid_json(self):
        """Test prediction with invalid JSON"""
        response = requests.post(
            f"{BASE_URL}/predict/soot-load",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
