# supplychain_api.py
# ---
# Simple API wrapper for the Supply Chain ML Pipeline
# This provides a clean interface for your backend team to integrate with
#
# Usage:
# python supplychain_api.py --model_path supplychain_model.pth
#

import json
import argparse
from typing import Dict, Any, List
from supplychain_ml_pipeline import SupplyChainMLPipeline

class SupplyChainAPI:
    """Simple API wrapper for supply chain risk prediction."""
    
    def __init__(self, model_path: str):
        """Initialize the API with a trained model."""
        self.pipeline = SupplyChainMLPipeline(model_path)
        print(f"Supply Chain API initialized with model: {model_path}")
    
    def predict_risk_batch(self, node_features: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Predict risk for multiple nodes.
        
        Args:
            node_features: Dict mapping node_id to feature dict
                Example: {
                    "N1": {
                        "news_count_1d": 2,
                        "news_count_7d": 8,
                        "neg_tone_frac_3d": 0.75,
                        "weather_anomaly_7d": 1,
                        "strike_flag_7d": 0,
                        "avg_lead_time_days": 21,
                        "inventory_days": 10,
                        "single_sourced": 0,
                        "past_delay_days": 7,
                        "news_velocity": 3.5
                    }
                }
        
        Returns:
            Dict with predictions and metadata
        """
        return self.pipeline.predict_risk(node_features)
    
    def predict_risk_single(self, node_id: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict risk for a single node.
        
        Args:
            node_id: Node identifier
            features: Feature dictionary for the node
        
        Returns:
            Dict with prediction for the single node
        """
        node_features = {node_id: features}
        results = self.pipeline.predict_risk(node_features)
        
        if node_id in results['predictions']:
            return {
                "node_id": node_id,
                "prediction": results['predictions'][node_id],
                "timestamp": results['timestamp']
            }
        else:
            return {"error": f"Node {node_id} not found or invalid"}
    
    def get_risk_explanation(self, node_id: str, prediction: Dict[str, Any]) -> str:
        """Get human-readable explanation for a risk prediction."""
        return self.pipeline.get_risk_explanation(node_id, prediction)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "feature_columns": self.pipeline.feature_columns,
            "num_nodes": len(self.pipeline.node_id_to_idx),
            "device": str(self.pipeline.device),
            "model_loaded": self.pipeline.model is not None
        }

def main():
    parser = argparse.ArgumentParser(description="Supply Chain Risk Prediction API")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for API server")
    
    args = parser.parse_args()
    
    # Initialize API
    api = SupplyChainAPI(args.model_path)
    
    print(f"Supply Chain Risk Prediction API")
    print(f"Model: {args.model_path}")
    print(f"Features: {api.get_model_info()['feature_columns']}")
    print(f"Available nodes: {list(api.pipeline.node_id_to_idx.keys())}")
    print()
    
    # Interactive mode for testing
    print("Interactive mode - Enter node features to get predictions")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            # Get node ID
            node_id = input("Enter node ID (or 'quit'): ").strip()
            if node_id.lower() == 'quit':
                break
            
            if node_id not in api.pipeline.node_id_to_idx:
                print(f"Error: Node {node_id} not found. Available nodes: {list(api.pipeline.node_id_to_idx.keys())}")
                continue
            
            # Get features
            print("Enter features (press Enter for default values):")
            features = {}
            for col in api.pipeline.feature_columns:
                value = input(f"  {col}: ").strip()
                if value:
                    try:
                        features[col] = float(value)
                    except ValueError:
                        print(f"Invalid value for {col}, using 0.0")
                        features[col] = 0.0
                else:
                    features[col] = 0.0
            
            # Predict
            result = api.predict_risk_single(node_id, features)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                prediction = result['prediction']
                print(f"\nPrediction for {node_id}:")
                print(f"  Risk Score: {prediction['risk_score']:.3f}")
                print(f"  Risk Level: {prediction['risk_level']}")
                print(f"  Confidence: {prediction['confidence']:.3f}")
                print(f"  Explanation: {api.get_risk_explanation(node_id, prediction)}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
