# Supply Chain Risk Prediction ML Pipeline

A production-ready machine learning pipeline for real-time supply chain risk prediction using Graph Neural Networks (GNNs).

## 🚀 Features

- **Real-time Risk Prediction**: Process live data updates and predict supply chain disruptions
- **Graph Neural Network**: Uses PyTorch Geometric for sophisticated graph-based learning
- **Model Persistence**: Save and load trained models for production deployment
- **Confidence Scoring**: Provides confidence levels for risk predictions
- **Human-readable Explanations**: Generate explanations for risk predictions
- **API Interface**: Clean interface for backend integration
- **Live Data Simulation**: Demo mode with simulated real-time data

## 📋 Requirements

```bash
pip install torch torch_geometric pandas scikit-learn numpy
```

## 🏗️ Architecture

```
Live Data Sources → Feature Engineering → GNN Model → Risk Prediction
     ↓                    ↓                ↓            ↓
News APIs          Feature Aggregation   PyTorch      Risk Scores
Weather APIs       (sliding windows)     Geometric    + Confidence
Strike Alerts      + Normalization       + MLP        + Explanations
```

## 📊 Data Format

### Node Features (CSV)
Required columns for each supply chain node:
- `node_id`: Unique identifier
- `news_count_1d`: News articles in last 1 day
- `news_count_7d`: News articles in last 7 days  
- `neg_tone_frac_3d`: Fraction of negative news in last 3 days
- `weather_anomaly_7d`: Binary flag for weather anomalies
- `strike_flag_7d`: Binary flag for labor strikes
- `avg_lead_time_days`: Average lead time in days
- `inventory_days`: Inventory coverage in days
- `single_sourced`: Binary flag for single-sourced suppliers
- `past_delay_days`: Past delay days
- `news_velocity`: News velocity metric
- `disruption_within_7d`: Target variable (0/1)

### Edge List (CSV)
- `src`: Source node ID
- `dst`: Destination node ID

## 🚀 Quick Start

### 1. Train a Model

```bash
python supplychain_ml_pipeline.py --train \
    --nodes_csv supplychain_node_features_example.csv \
    --edges_csv supplychain_edges_example.csv \
    --epochs 200 \
    --lr 0.01
```

### 2. Run Live Inference

```bash
python supplychain_ml_pipeline.py --infer \
    --model_path supplychain_model.pth \
    --simulate_live
```

### 3. Interactive API

```bash
python supplychain_api.py --model_path supplychain_model.pth
```

## 🔧 API Usage

### Python Integration

```python
from supplychain_ml_pipeline import SupplyChainMLPipeline

# Initialize pipeline
pipeline = SupplyChainMLPipeline("supplychain_model.pth")

# Live data update
live_features = {
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

# Get predictions
results = pipeline.predict_risk(live_features)
print(results)
```

### API Wrapper

```python
from supplychain_api import SupplyChainAPI

# Initialize API
api = SupplyChainAPI("supplychain_model.pth")

# Single node prediction
result = api.predict_risk_single("N1", features)
print(f"Risk Level: {result['prediction']['risk_level']}")

# Batch prediction
results = api.predict_risk_batch({"N1": features, "N2": features2})
```

## 📈 Model Architecture

The GNN model consists of:

1. **Two GCN Layers**: Graph Convolutional Networks for learning node representations
2. **Batch Normalization**: Stabilizes training
3. **Dropout**: Prevents overfitting
4. **MLP Head**: Final prediction layer with sigmoid activation

```
Input Features → GCN1 → BatchNorm → ReLU → Dropout
                     ↓
                GCN2 → BatchNorm → ReLU → Dropout
                     ↓
                MLP → Sigmoid → Risk Score [0,1]
```

## 🎯 Risk Classification

- **LOW**: Risk score < 0.3
- **MEDIUM**: Risk score 0.3 - 0.7  
- **HIGH**: Risk score > 0.7

## 🔄 Live Data Integration

### For Your Backend Team

1. **Data Ingestion**: Collect features from:
   - News APIs (GDELT, NewsAPI)
   - Weather APIs (OpenWeatherMap)
   - Labor/strike monitoring systems
   - Internal supply chain data

2. **Feature Engineering**: Aggregate into sliding windows:
   ```python
   # Example feature update
   features = {
       "news_count_1d": count_news_last_24h(),
       "neg_tone_frac_3d": calculate_negative_tone_ratio(),
       "weather_anomaly_7d": check_weather_alerts(),
       # ... other features
   }
   ```

3. **ML Integration**: Call the prediction API:
   ```python
   # Update node features
   pipeline.update_node_features("N1", features)
   
   # Get predictions
   predictions = pipeline.predict_risk(live_features)
   ```

## 📊 Example Output

```
REAL-TIME SUPPLY CHAIN RISK PREDICTIONS
============================================================
Timestamp: 2025-09-19T21:32:48.448568
Total nodes analyzed: 3

Node: N1
  Risk Score: 1.000
  Risk Level: HIGH
  Confidence: 1.000
  Explanation: Node N1 flagged as HIGH RISK (score: 1.00) due to: 
               High news volume (8 articles in 7 days), 
               Negative news tone (75.0% negative), 
               Weather anomaly detected

Node: N2
  Risk Score: 0.997
  Risk Level: HIGH
  Confidence: 0.995
  Explanation: Node N2 flagged as HIGH RISK (score: 1.00) due to: 
               Labor strike reported, 
               Single-sourced supplier (high dependency)
```

## 🛠️ Production Deployment

### Model Training
- Train on historical data with known disruptions
- Use cross-validation for robust evaluation
- Save model artifacts for deployment

### Live Inference
- Deploy model as a service
- Set up feature pipelines for real-time updates
- Monitor model performance and retrain as needed

### Scaling Considerations
- For large graphs, consider graph sampling techniques
- Implement caching for frequently accessed predictions
- Use batch processing for multiple node updates

## 🔧 Configuration

### Training Parameters
- `--epochs`: Number of training epochs (default: 200)
- `--lr`: Learning rate (default: 0.01)
- `--weight_decay`: L2 regularization (default: 1e-4)
- `--val_frac`: Validation fraction (default: 0.3)

### Model Architecture
- `hidden`: Hidden dimension size (default: 64)
- `dropout`: Dropout rate (default: 0.3)

## 📝 Notes

- The current implementation uses a simplified approach for live predictions
- For production, implement proper subgraph handling for large graphs
- Consider adding more sophisticated feature engineering
- Monitor model drift and implement retraining pipelines

## 🤝 Integration with Your Team

This ML pipeline provides:
- **Clean API interface** for backend integration
- **Model persistence** for deployment
- **Real-time inference** capabilities
- **Confidence scoring** for decision making
- **Human-readable explanations** for stakeholders

Your backend team can integrate this by:
1. Calling the prediction API with live feature updates
2. Storing model artifacts in your deployment pipeline
3. Setting up monitoring and alerting based on risk scores
4. Implementing the feature engineering pipeline for live data sources