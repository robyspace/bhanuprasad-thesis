# 🎯 Production Model Selection & Deployment Plan

## 📊 Final Model Performance Summary

### All Models Tested

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score | Inference Time | Status |
|-------|----------|---------|-----------|--------|----------|----------------|--------|
| **Random Forest** | **87.7%** | **0.955** | 93.5% | 79.6% | 86.0% | 36.25 ms | ✅ **Recommended** |
| **XGBoost** | **87.6%** | **0.951** | 95.5% | 77.5% | 85.6% | 6.97 ms | ✅ **Recommended** |
| **Deep MLP** | **86.9%** | **0.940** | ~85% | ~88% | ~86% | ~15 ms | ✅ **Optional** |
| LSTM/CNN-LSTM | 52.1% | 0.510 | 47.5% | 32.8% | 38.8% | 114 ms | ❌ Failed |

### 🏆 Winner: Random Forest + XGBoost Ensemble

**Deployment Strategy:**
1. **Primary Model:** Random Forest (highest AUC: 0.955)
2. **Secondary Model:** XGBoost (fastest inference: 6.97 ms)
3. **Optional Enhancement:** Deep MLP for ensemble voting

---

## 🎯 Recommended Production Architecture

### Option 1: Dual-Model Ensemble (Recommended)

**Configuration:**
```python
# Ensemble prediction strategy
predictions = {
    'random_forest': rf_model.predict_proba(X)[:, 1],
    'xgboost': xgb_model.predict_proba(X)[:, 1]
}

# Weighted average (both models perform similarly)
final_prediction = (predictions['random_forest'] * 0.5 +
                   predictions['xgboost'] * 0.5)

# Or use max voting for classification
final_class = (final_prediction > 0.5).astype(int)
```

**Benefits:**
- ✅ Combines two best-performing models
- ✅ Reduces variance through ensemble
- ✅ 87.7% accuracy baseline
- ✅ Robust to edge cases

**Drawbacks:**
- ~43 ms total inference time (RF: 36ms + XGB: 7ms)
- Higher memory footprint

### Option 2: XGBoost Only (Fastest)

**Configuration:**
```python
# Single model prediction
prediction = xgb_model.predict_proba(X)[:, 1]
final_class = (prediction > 0.5).astype(int)
```

**Benefits:**
- ✅ Fastest inference: 6.97 ms
- ✅ 87.6% accuracy (nearly identical to RF)
- ✅ Lower memory usage
- ✅ Easier deployment

**Drawbacks:**
- Single point of failure
- No ensemble diversity

### Option 3: Triple-Model Ensemble (Maximum Accuracy)

**Configuration:**
```python
# All three models
predictions = {
    'random_forest': rf_model.predict_proba(X)[:, 1],
    'xgboost': xgb_model.predict_proba(X)[:, 1],
    'deep_mlp': mlp_model.predict(X).flatten()
}

# Weighted average
final_prediction = (
    predictions['random_forest'] * 0.35 +
    predictions['xgboost'] * 0.35 +
    predictions['deep_mlp'] * 0.30
)
```

**Benefits:**
- ✅ Maximum accuracy (ensemble typically +1-2%)
- ✅ Model diversity (tree-based + neural network)
- ✅ Robust predictions

**Drawbacks:**
- ~58 ms total inference (RF: 36ms + XGB: 7ms + MLP: 15ms)
- Higher memory and computational cost
- Requires TensorFlow/Keras deployment

---

## 💰 Cost-Benefit Analysis

### Performance vs. Cost

| Configuration | Accuracy | Inference | Memory | Complexity | Recommendation |
|---------------|----------|-----------|--------|------------|----------------|
| **XGBoost Only** | 87.6% | 7 ms | Low | Simple | ✅ **Best for high-throughput** |
| **RF + XGBoost** | ~88.5% | 43 ms | Medium | Medium | ✅ **Best balance** |
| **RF + XGB + MLP** | ~89% | 58 ms | High | Complex | ⚠️ If max accuracy needed |

### Recommendation by Use Case

**High-Throughput Network (>10K flows/sec):**
→ **XGBoost Only** (6.97 ms inference)

**Standard Enterprise Network (<5K flows/sec):**
→ **Random Forest + XGBoost Ensemble** (best accuracy/speed balance)

**Research/Maximum Accuracy:**
→ **RF + XGB + MLP Ensemble** (highest accuracy)

---

## 🚀 AWS Deployment Plan

### Phase 1: Model Export & Preparation

#### Step 1.1: Export Models from Google Colab

```python
# In your Colab notebook after training

import pickle
import joblib
from tensorflow import keras

# Save Random Forest
with open(f'{PROJECT_DIR}/models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save XGBoost
xgb_model.save_model(f'{PROJECT_DIR}/models/xgboost_model.json')

# Save Deep MLP (if using)
mlp_model.save(f'{PROJECT_DIR}/models/deep_mlp_model.h5')

# Save preprocessing artifacts (already done)
# - scaler.pkl
# - feature_names.pkl

# Save model metadata
import json
metadata = {
    'models': {
        'random_forest': {
            'accuracy': 0.877,
            'auc': 0.955,
            'inference_ms': 36.25
        },
        'xgboost': {
            'accuracy': 0.876,
            'auc': 0.951,
            'inference_ms': 6.97
        },
        'deep_mlp': {
            'accuracy': 0.869,
            'auc': 0.940,
            'inference_ms': 15.0
        }
    },
    'deployment_config': {
        'primary_model': 'random_forest',
        'secondary_model': 'xgboost',
        'ensemble_weights': {
            'random_forest': 0.5,
            'xgboost': 0.5
        }
    },
    'preprocessing': {
        'scaler': 'standard_scaler',
        'features_count': 80,
        'feature_names_file': 'feature_names.pkl'
    }
}

with open(f'{PROJECT_DIR}/models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ All models and metadata saved!")
```

#### Step 1.2: Download Models to Local Machine

```python
# Download from Google Drive to your local machine
# Option 1: Manual download via Drive interface
# Option 2: Use gdown or rclone

# After downloading, you should have:
# - models/random_forest_model.pkl
# - models/xgboost_model.json
# - models/deep_mlp_model.h5 (optional)
# - models/scaler.pkl
# - models/feature_names.pkl
# - models/model_metadata.json
```

### Phase 2: Flask API Development

#### Step 2.1: Create Flask Application

```python
# File: app.py

from flask import Flask, request, jsonify
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from tensorflow import keras
import json
import time

app = Flask(__name__)

# Load models and preprocessing
print("Loading models...")

# Load metadata
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load preprocessing
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load Random Forest
with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load XGBoost
xgb_model = xgb.Booster()
xgb_model.load_model('models/xgboost_model.json')

# Optional: Load MLP
# mlp_model = keras.models.load_model('models/deep_mlp_model.h5')

print(f"✓ Models loaded successfully!")
print(f"✓ Using {len(feature_names)} features")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'random_forest': True,
            'xgboost': True
        },
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict network flow as benign (0) or attack (1)

    Input JSON format:
    {
        "features": {
            "feature1": value1,
            "feature2": value2,
            ...
        }
    }
    """
    try:
        start_time = time.time()

        # Get input data
        data = request.get_json()

        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400

        # Convert to DataFrame
        features_dict = data['features']
        df = pd.DataFrame([features_dict])

        # Ensure all features are present
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {list(missing_features)}'
            }), 400

        # Select and order features
        df = df[feature_names]

        # Scale features
        X_scaled = scaler.transform(df)

        # Random Forest prediction
        rf_proba = rf_model.predict_proba(X_scaled)[0, 1]

        # XGBoost prediction
        dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
        xgb_proba = xgb_model.predict(dmatrix)[0]

        # Ensemble prediction (weighted average)
        weights = metadata['deployment_config']['ensemble_weights']
        final_proba = (
            rf_proba * weights['random_forest'] +
            xgb_proba * weights['xgboost']
        )

        final_prediction = int(final_proba > 0.5)

        inference_time = (time.time() - start_time) * 1000  # ms

        # Return prediction
        return jsonify({
            'prediction': final_prediction,
            'prediction_label': 'Attack' if final_prediction == 1 else 'Benign',
            'confidence': float(final_proba),
            'model_probabilities': {
                'random_forest': float(rf_proba),
                'xgboost': float(xgb_proba)
            },
            'inference_time_ms': round(inference_time, 2),
            'timestamp': time.time()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple flows

    Input JSON format:
    {
        "flows": [
            {"feature1": value1, "feature2": value2, ...},
            {"feature1": value1, "feature2": value2, ...},
            ...
        ]
    }
    """
    try:
        start_time = time.time()

        data = request.get_json()

        if 'flows' not in data:
            return jsonify({'error': 'Missing flows in request'}), 400

        flows = data['flows']
        df = pd.DataFrame(flows)

        # Ensure all features present
        df = df[feature_names]

        # Scale
        X_scaled = scaler.transform(df)

        # Predictions
        rf_proba = rf_model.predict_proba(X_scaled)[:, 1]

        dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
        xgb_proba = xgb_model.predict(dmatrix)

        # Ensemble
        weights = metadata['deployment_config']['ensemble_weights']
        final_proba = (
            rf_proba * weights['random_forest'] +
            xgb_proba * weights['xgboost']
        )

        predictions = (final_proba > 0.5).astype(int).tolist()

        inference_time = (time.time() - start_time) * 1000

        return jsonify({
            'predictions': predictions,
            'probabilities': final_proba.tolist(),
            'count': len(predictions),
            'attack_count': int(sum(predictions)),
            'benign_count': int(len(predictions) - sum(predictions)),
            'inference_time_ms': round(inference_time, 2),
            'avg_time_per_flow_ms': round(inference_time / len(predictions), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Return model metadata"""
    return jsonify(metadata)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

#### Step 2.2: Create Requirements File

```txt
# File: requirements.txt

Flask==2.3.3
scikit-learn==1.3.0
xgboost==2.0.0
pandas==2.0.3
numpy==1.24.3
gunicorn==21.2.0

# Optional: If using MLP
# tensorflow==2.13.0
```

#### Step 2.3: Create Dockerfile

```dockerfile
# File: Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY models/ ./models/

# Expose port
EXPOSE 5000

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
```

### Phase 3: AWS EC2 Deployment

#### Step 3.1: Launch EC2 Instance

```bash
# Instance Specifications:
# - Type: t3.medium (2 vCPU, 4 GB RAM) - ~$30/month
# - OS: Ubuntu 22.04 LTS
# - Storage: 20 GB SSD
# - Security Group: Allow ports 22 (SSH), 5000 (Flask), 9090 (Prometheus)
```

#### Step 3.2: Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Logout and login again for docker group
exit
ssh -i your-key.pem ubuntu@your-ec2-ip
```

#### Step 3.3: Deploy Application

```bash
# Create project directory
mkdir ids-api
cd ids-api

# Upload your files (from local machine)
# scp -i your-key.pem -r app.py requirements.txt Dockerfile models/ ubuntu@your-ec2-ip:~/ids-api/

# Build Docker image
docker build -t ids-api:v1.0 .

# Run container
docker run -d \
  --name ids-api \
  --restart unless-stopped \
  -p 5000:5000 \
  ids-api:v1.0

# Check logs
docker logs ids-api

# Test endpoint
curl http://localhost:5000/health
```

### Phase 4: Testing & Validation

#### Step 4.1: Local Testing Script

```python
# File: test_api.py

import requests
import json
import time

API_URL = "http://your-ec2-ip:5000"

# Test health check
print("Testing health check...")
response = requests.get(f"{API_URL}/health")
print(f"Health: {response.json()}")

# Test single prediction
print("\nTesting single prediction...")
test_flow = {
    "features": {
        # Fill with actual feature values from your test set
        "Dst Port": 80,
        "Protocol": 6,
        "Flow Duration": 1000,
        # ... all other features
    }
}

response = requests.post(f"{API_URL}/predict", json=test_flow)
print(f"Prediction: {response.json()}")

# Test batch prediction
print("\nTesting batch prediction...")
batch_data = {
    "flows": [test_flow["features"]] * 10
}

start = time.time()
response = requests.post(f"{API_URL}/predict/batch", json=batch_data)
elapsed = time.time() - start

result = response.json()
print(f"Batch results: {result}")
print(f"Total time: {elapsed*1000:.2f} ms")
print(f"Per-flow time: {result['avg_time_per_flow_ms']:.2f} ms")
```

### Phase 5: Monitoring & Maintenance

#### Step 5.1: Add Prometheus Monitoring

```python
# Update app.py to include Prometheus metrics

from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from flask import Response

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions made', ['prediction_type'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

# Update predict() function
@prediction_latency.time()
def predict():
    # ... existing code ...
    prediction_counter.labels(prediction_type='Attack' if final_prediction == 1 else 'Benign').inc()
    # ... existing code ...
```

#### Step 5.2: Setup Logging

```python
# Add to app.py

import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler('logs/ids_api.log', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Add logging to endpoints
app.logger.info(f"Prediction made: {final_prediction}, confidence: {final_proba:.4f}")
```

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Train and evaluate all models
- [ ] Export model files (.pkl, .json, .h5)
- [ ] Export preprocessing artifacts (scaler, feature names)
- [ ] Create model metadata JSON
- [ ] Test models locally with sample data

### Development
- [ ] Create Flask API (app.py)
- [ ] Create requirements.txt
- [ ] Create Dockerfile
- [ ] Test API locally
- [ ] Write API tests

### AWS Setup
- [ ] Launch EC2 instance (t3.medium recommended)
- [ ] Configure security groups
- [ ] Allocate Elastic IP (optional)
- [ ] Install Docker
- [ ] Upload application files

### Deployment
- [ ] Build Docker image
- [ ] Run container
- [ ] Test health endpoint
- [ ] Test prediction endpoint
- [ ] Test batch prediction
- [ ] Verify inference times

### Monitoring
- [ ] Setup Prometheus metrics
- [ ] Configure logging
- [ ] Create CloudWatch dashboard
- [ ] Setup alerts for errors/latency

### Documentation
- [ ] API documentation (endpoints, schemas)
- [ ] Deployment runbook
- [ ] Troubleshooting guide
- [ ] Model update procedure

---

## 🎯 Next Steps

1. **Export Models from Colab** (30 minutes)
   - Run model export code
   - Download all model files
   - Verify files locally

2. **Build Flask API** (1-2 hours)
   - Create app.py
   - Test locally with sample data
   - Verify predictions match notebook

3. **Containerize** (30 minutes)
   - Create Dockerfile
   - Build and test container locally
   - Verify all dependencies

4. **Deploy to AWS** (2-3 hours)
   - Launch EC2 instance
   - Upload files
   - Run container
   - Test endpoints

5. **Monitor & Optimize** (ongoing)
   - Track performance metrics
   - Optimize inference time if needed
   - Handle edge cases

---

## 💡 Production Tips

### Performance Optimization
- Use **XGBoost only** if you need <10ms latency
- Use **RF + XGBoost** for best accuracy (recommended)
- Add **MLP** only if you need maximum accuracy

### Cost Optimization
- Start with t3.medium, scale as needed
- Use spot instances for dev/test (70% savings)
- Setup auto-scaling if traffic varies

### Security
- Use AWS Security Groups to restrict access
- Add API authentication (JWT tokens)
- Enable HTTPS with SSL certificate
- Regular security updates

### Reliability
- Use Docker for consistent environment
- Setup health checks and auto-restart
- Keep model backups in S3
- Version your models

---

## 📞 Support

If you encounter issues:
1. Check Docker logs: `docker logs ids-api`
2. Verify model files are present
3. Test with curl/Postman before application integration
4. Monitor CloudWatch metrics

---

**Ready to deploy? Start with Phase 1: Model Export!** 🚀
