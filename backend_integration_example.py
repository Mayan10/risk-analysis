# backend_integration_example.py
# ---
# Example showing how your backend team can integrate the ML pipeline
# This simulates a real backend system with live data updates

import json
import time
from datetime import datetime
from typing import Dict, Any
from supplychain_ml_pipeline import SupplyChainMLPipeline

class SupplyChainBackend:
    """Example backend system that integrates with the ML pipeline."""
    
    def __init__(self, model_path: str):
        """Initialize backend with ML pipeline."""
        self.ml_pipeline = SupplyChainMLPipeline(model_path)
        self.node_features = {}  # In production, this would be a database
        self.risk_alerts = []    # Store risk alerts
        
    def update_node_features(self, node_id: str, features: Dict[str, float]) -> None:
        """Update features for a node (simulates live data ingestion)."""
        self.node_features[node_id] = {
            **features,
            "last_updated": datetime.now().isoformat()
        }
        print(f"Updated features for {node_id}: {features}")
    
    def get_risk_predictions(self, node_ids: list = None) -> Dict[str, Any]:
        """Get risk predictions for specified nodes or all nodes."""
        if node_ids is None:
            node_ids = list(self.node_features.keys())
        
        # Filter features for requested nodes
        features_subset = {
            node_id: features for node_id, features in self.node_features.items()
            if node_id in node_ids
        }
        
        if not features_subset:
            return {"error": "No features available for requested nodes"}
        
        # Get ML predictions
        predictions = self.ml_pipeline.predict_risk(features_subset)
        
        # Check for high-risk alerts
        for node_id, pred in predictions.get('predictions', {}).items():
            if pred['risk_level'] == 'HIGH' and pred['confidence'] > 0.8:
                alert = {
                    "node_id": node_id,
                    "risk_score": pred['risk_score'],
                    "explanation": self.ml_pipeline.get_risk_explanation(node_id, pred),
                    "timestamp": datetime.now().isoformat()
                }
                self.risk_alerts.append(alert)
                print(f"🚨 HIGH RISK ALERT: {alert}")
        
        return predictions
    
    def get_risk_alerts(self) -> list:
        """Get all risk alerts."""
        return self.risk_alerts
    
    def simulate_live_data_updates(self):
        """Simulate live data updates (for demo purposes)."""
        print("🔄 Simulating live data updates...")
        
        # Simulate different scenarios
        scenarios = [
            {
                "N1": {
                    "news_count_1d": 3,
                    "news_count_7d": 12,
                    "neg_tone_frac_3d": 0.8,
                    "weather_anomaly_7d": 1,
                    "strike_flag_7d": 0,
                    "avg_lead_time_days": 21,
                    "inventory_days": 8,
                    "single_sourced": 0,
                    "past_delay_days": 7,
                    "news_velocity": 4.2
                }
            },
            {
                "N2": {
                    "news_count_1d": 1,
                    "news_count_7d": 2,
                    "neg_tone_frac_3d": 0.1,
                    "weather_anomaly_7d": 0,
                    "strike_flag_7d": 1,
                    "avg_lead_time_days": 10,
                    "inventory_days": 25,
                    "single_sourced": 1,
                    "past_delay_days": 7,
                    "news_velocity": 1.5
                }
            },
            {
                "N3": {
                    "news_count_1d": 0,
                    "news_count_7d": 1,
                    "neg_tone_frac_3d": 0.0,
                    "weather_anomaly_7d": 0,
                    "strike_flag_7d": 0,
                    "avg_lead_time_days": 7,
                    "inventory_days": 15,
                    "single_sourced": 1,
                    "past_delay_days": 2,
                    "news_velocity": 0.3
                }
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            print(f"\n--- Scenario {i+1} ---")
            
            # Update features
            for node_id, features in scenario.items():
                self.update_node_features(node_id, features)
            
            # Get predictions
            predictions = self.get_risk_predictions()
            
            # Display results
            print(f"\n📊 Risk Predictions:")
            for node_id, pred in predictions.get('predictions', {}).items():
                print(f"  {node_id}: {pred['risk_level']} (score: {pred['risk_score']:.3f})")
            
            time.sleep(1)  # Simulate time delay

def main():
    """Example usage of the backend integration."""
    print("🏭 Supply Chain Backend Integration Example")
    print("=" * 50)
    
    # Initialize backend
    backend = SupplyChainBackend("supplychain_model.pth")
    
    # Simulate live data updates and predictions
    backend.simulate_live_data_updates()
    
    # Show all alerts
    alerts = backend.get_risk_alerts()
    if alerts:
        print(f"\n🚨 Total Risk Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert['node_id']}: {alert['explanation']}")
    else:
        print("\n✅ No high-risk alerts generated")
    
    print("\n🎯 Integration Complete!")
    print("\nFor your backend team:")
    print("1. Replace simulate_live_data_updates() with real data ingestion")
    print("2. Connect to your database for feature storage")
    print("3. Implement alerting/notification system")
    print("4. Add monitoring and logging")

if __name__ == "__main__":
    main()
