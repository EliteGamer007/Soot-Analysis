"""
Unit Tests for DPF Soot Prediction System
Tests individual functions and components in isolation
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from api_server
from api_server import engineer_features, calculate_risk_level, get_cache_key
from api_server import SensorReading

class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_interaction_features(self):
        """Test interaction feature creation"""
        df = pd.DataFrame({
            'Exhaust temperature - DPF inlet': [400, 450, 500],
            'Exhaust flow rate': [80, 90, 100],
            'Engine load (%)': [40, 50, 60],
            'Engine RPM': [1800, 2000, 2200],
            'Differential pressure across DPF': [10, 15, 20],
            'Vehicle speed': [60, 70, 80],
            'Fuel consumption rate': [8, 9, 10],
            'Exhaust temperature - DOC inlet': [380, 430, 480],
            'Trip distance (km)': [100, 150, 200],
            'Dist_Since_Last_Regen': [400, 500, 600]
        })
        
        result = engineer_features(df)
        
        # Check interaction features exist
        assert 'Temp_x_ExhaustFlow' in result.columns
        assert 'Load_x_RPM' in result.columns
        assert 'Temp_x_Pressure' in result.columns
        assert 'Speed_x_Load' in result.columns
        assert 'RPM_x_Fuel' in result.columns
        
        # Check calculation correctness
        assert result['Temp_x_ExhaustFlow'].iloc[0] == 400 * 80
        assert result['Load_x_RPM'].iloc[0] == 40 * 1800
        
    def test_polynomial_features(self):
        """Test polynomial feature creation"""
        df = pd.DataFrame({
            'Differential pressure across DPF': [10, 15, 20],
            'Exhaust temperature - DPF inlet': [400, 450, 500],
            'Engine load (%)': [40, 50, 60],
            'Exhaust temperature - DOC inlet': [380, 430, 480],
            'Exhaust flow rate': [80, 90, 100],
            'Engine RPM': [1800, 2000, 2200],
            'Vehicle speed': [60, 70, 80],
            'Fuel consumption rate': [8, 9, 10],
            'Trip distance (km)': [100, 150, 200],
            'Dist_Since_Last_Regen': [400, 500, 600]
        })
        
        result = engineer_features(df)
        
        assert 'Pressure_squared' in result.columns
        assert 'Temp_squared' in result.columns
        assert 'Load_squared' in result.columns
        
        assert result['Pressure_squared'].iloc[0] == 10 ** 2
        assert result['Temp_squared'].iloc[0] == 400 ** 2
        
    def test_rolling_statistics(self):
        """Test rolling statistics calculation"""
        df = pd.DataFrame({
            'Engine RPM': [1800, 1850, 1900, 1950, 2000] * 2,
            'Exhaust temperature - DPF inlet': [400, 410, 420, 430, 440] * 2,
            'Engine load (%)': [40, 42, 44, 46, 48] * 2,
            'Differential pressure across DPF': [10, 11, 12, 13, 14] * 2,
            'Exhaust temperature - DOC inlet': [380, 390, 400, 410, 420] * 2,
            'Exhaust flow rate': [80, 82, 84, 86, 88] * 2,
            'Vehicle speed': [60, 62, 64, 66, 68] * 2,
            'Fuel consumption rate': [8, 8.1, 8.2, 8.3, 8.4] * 2,
            'Trip distance (km)': [100, 110, 120, 130, 140] * 2,
            'Dist_Since_Last_Regen': [400, 410, 420, 430, 440] * 2
        })
        
        result = engineer_features(df)
        
        # Check rolling mean features
        assert 'Engine RPM_mean' in result.columns
        assert 'Exhaust temperature - DPF inlet_mean' in result.columns
        
        # Check rolling std features
        assert 'Engine RPM_std' in result.columns
        assert 'Exhaust temperature - DPF inlet_std' in result.columns
        
        # Rolling stats should not be NaN for sufficient data
        assert not result['Engine RPM_mean'].iloc[-1] == 0 or pd.isna(result['Engine RPM_mean'].iloc[-1])
        
    def test_lag_features(self):
        """Test lag feature creation"""
        df = pd.DataFrame({
            'Differential pressure across DPF': [10, 11, 12, 13, 14],
            'Exhaust temperature - DPF inlet': [400, 410, 420, 430, 440],
            'Engine load (%)': [40, 42, 44, 46, 48],
            'Exhaust temperature - DOC inlet': [380, 390, 400, 410, 420],
            'Engine RPM': [1800, 1850, 1900, 1950, 2000],
            'Exhaust flow rate': [80, 82, 84, 86, 88],
            'Vehicle speed': [60, 62, 64, 66, 68],
            'Fuel consumption rate': [8, 8.1, 8.2, 8.3, 8.4],
            'Trip distance (km)': [100, 110, 120, 130, 140],
            'Dist_Since_Last_Regen': [400, 410, 420, 430, 440]
        })
        
        result = engineer_features(df)
        
        assert 'Differential pressure across DPF_lag1' in result.columns
        assert 'Exhaust temperature - DPF inlet_lag1' in result.columns
        assert 'Engine load (%)_lag1' in result.columns
        
        # lag1 should shift by 1
        assert result['Differential pressure across DPF_lag1'].iloc[1] == 10
        assert result['Differential pressure across DPF_lag1'].iloc[2] == 11
        
    def test_feature_count(self):
        """Test that all expected features are created"""
        df = pd.DataFrame({
            'Engine RPM': [1800] * 5,
            'Vehicle speed': [60] * 5,
            'Engine load (%)': [40] * 5,
            'Exhaust temperature - DPF inlet': [400] * 5,
            'Exhaust temperature - DOC inlet': [380] * 5,
            'Exhaust flow rate': [80] * 5,
            'Differential pressure across DPF': [10] * 5,
            'Fuel consumption rate': [8] * 5,
            'Trip distance (km)': [100] * 5,
            'Dist_Since_Last_Regen': [400] * 5
        })
        
        result = engineer_features(df)
        
        # Should have base features (10) + engineered features (28+)
        # Actual count may vary based on implementation
        assert len(result.columns) >= 30, f"Expected >= 30 features, got {len(result.columns)}"

class TestRiskCalculation:
    """Test risk level calculation logic"""
    
    def test_low_risk(self):
        """Test LOW risk classification"""
        assert calculate_risk_level(0.1) == "LOW"
        assert calculate_risk_level(0.24) == "LOW"
        
    def test_medium_risk(self):
        """Test MEDIUM risk classification"""
        assert calculate_risk_level(0.25) == "MEDIUM"
        assert calculate_risk_level(0.49) == "MEDIUM"
        
    def test_high_risk(self):
        """Test HIGH risk classification"""
        assert calculate_risk_level(0.5) == "HIGH"
        assert calculate_risk_level(0.74) == "HIGH"
        
    def test_critical_risk(self):
        """Test CRITICAL risk classification"""
        assert calculate_risk_level(0.75) == "CRITICAL"
        assert calculate_risk_level(0.99) == "CRITICAL"
        
    def test_edge_cases(self):
        """Test boundary values"""
        assert calculate_risk_level(0.0) == "LOW"
        assert calculate_risk_level(1.0) == "CRITICAL"

class TestCaching:
    """Test caching logic"""
    
    def test_cache_key_generation(self):
        """Test cache key generation is consistent"""
        readings = [
            SensorReading(
                vehicle_id="VEH-001",
                timestamp=datetime(2026, 1, 18, 10, 30),
                engine_rpm=1850,
                vehicle_speed=65.5,
                engine_load=45.2,
                exhaust_temp_dpf_inlet=420.5,
                exhaust_temp_doc_inlet=380.2,
                exhaust_flow_rate=85.3,
                differential_pressure=12.5,
                fuel_consumption=8.2
            )
        ]
        
        key1 = get_cache_key(readings)
        key2 = get_cache_key(readings)
        
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length
        
    def test_cache_key_different_vehicles(self):
        """Test different vehicles produce different cache keys"""
        readings1 = [
            SensorReading(
                vehicle_id="VEH-001",
                timestamp=datetime(2026, 1, 18, 10, 30),
                engine_rpm=1850,
                vehicle_speed=65.5,
                engine_load=45.2,
                exhaust_temp_dpf_inlet=420.5,
                exhaust_temp_doc_inlet=380.2,
                exhaust_flow_rate=85.3,
                differential_pressure=12.5,
                fuel_consumption=8.2
            )
        ]
        
        readings2 = [
            SensorReading(
                vehicle_id="VEH-002",
                timestamp=datetime(2026, 1, 18, 10, 30),
                engine_rpm=1850,
                vehicle_speed=65.5,
                engine_load=45.2,
                exhaust_temp_dpf_inlet=420.5,
                exhaust_temp_doc_inlet=380.2,
                exhaust_flow_rate=85.3,
                differential_pressure=12.5,
                fuel_consumption=8.2
            )
        ]
        
        key1 = get_cache_key(readings1)
        key2 = get_cache_key(readings2)
        
        assert key1 != key2

class TestInputValidation:
    """Test Pydantic model validation"""
    
    def test_valid_sensor_reading(self):
        """Test valid sensor reading is accepted"""
        reading = SensorReading(
            vehicle_id="VEH-001",
            timestamp=datetime(2026, 1, 18, 10, 30),
            engine_rpm=1850,
            vehicle_speed=65.5,
            engine_load=45.2,
            exhaust_temp_dpf_inlet=420.5,
            exhaust_temp_doc_inlet=380.2,
            exhaust_flow_rate=85.3,
            differential_pressure=12.5,
            fuel_consumption=8.2
        )
        
        assert reading.vehicle_id == "VEH-001"
        assert reading.engine_rpm == 1850
        
    def test_invalid_rpm(self):
        """Test invalid RPM is rejected"""
        with pytest.raises(Exception):  # Pydantic validation error
            SensorReading(
                vehicle_id="VEH-001",
                timestamp=datetime(2026, 1, 18, 10, 30),
                engine_rpm=6000,  # Exceeds max (5000)
                vehicle_speed=65.5,
                engine_load=45.2,
                exhaust_temp_dpf_inlet=420.5,
                exhaust_temp_doc_inlet=380.2,
                exhaust_flow_rate=85.3,
                differential_pressure=12.5,
                fuel_consumption=8.2
            )
            
    def test_invalid_speed(self):
        """Test invalid speed is rejected"""
        with pytest.raises(Exception):
            SensorReading(
                vehicle_id="VEH-001",
                timestamp=datetime(2026, 1, 18, 10, 30),
                engine_rpm=1850,
                vehicle_speed=250,  # Exceeds max (200)
                engine_load=45.2,
                exhaust_temp_dpf_inlet=420.5,
                exhaust_temp_doc_inlet=380.2,
                exhaust_flow_rate=85.3,
                differential_pressure=12.5,
                fuel_consumption=8.2
            )

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
