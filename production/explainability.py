"""
Model Explainability Module - SHAP Integration
Provides interpretable predictions with feature importance
"""
import shap
import numpy as np
import pandas as pd
from typing import Dict, List

class ModelExplainer:
    """Generates SHAP explanations for DPF soot predictions"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def initialize_explainer(self, X_background: pd.DataFrame):
        """
        Initialize SHAP explainer with background data
        Background data should be a sample of training data (100-1000 rows)
        """
        self.explainer = shap.TreeExplainer(
            self.model,
            X_background,
            feature_names=self.feature_names
        )
        
    def explain_prediction(self, X: pd.DataFrame) -> Dict:
        """
        Generate SHAP explanation for a single prediction
        
        Returns:
            Dictionary with feature contributions and base value
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first")
            
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, use positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
            
        # Get top contributing features
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0],
            'abs_shap': np.abs(shap_values[0])
        }).sort_values('abs_shap', ascending=False)
        
        return {
            'base_value': float(self.explainer.expected_value),
            'prediction_value': float(self.explainer.expected_value + shap_values[0].sum()),
            'top_features': feature_importance.head(10).to_dict('records'),
            'all_shap_values': shap_values[0].tolist()
        }
    
    def get_feature_importance(self, X: pd.DataFrame) -> Dict:
        """Get global feature importance across dataset"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
            
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        return {
            'feature_importance': importance_df.to_dict('records'),
            'top_5_features': importance_df.head(5)['feature'].tolist()
        }

# Example usage in API:
"""
# Initialize once at startup
from api_server import model_state
explainer = ModelExplainer(model_state.model, model_state.features)

# Load background data (sample from training)
X_background = pd.read_csv('data/background_sample.csv')[model_state.features]
explainer.initialize_explainer(X_background)

# In prediction endpoint:
explanation = explainer.explain_prediction(X)

# Return enhanced response:
{
    "soot_load_percent": 75.2,
    "risk_level": "HIGH",
    "explanation": {
        "top_contributors": explanation['top_features'][:5],
        "why_high_risk": "High differential pressure (35 kPa) and long distance since regen (650 km)"
    }
}
"""
