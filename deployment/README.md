# 🚀 IDS API Deployment Guide

Production-ready Flask API for Network Intrusion Detection using ensemble of Random Forest + XGBoost models.

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | Inference Time |
|-------|----------|---------|----------------|
| Random Forest | 87.7% | 0.955 | 36.25 ms |
| XGBoost | 87.6% | 0.951 | 6.97 ms |
| **Ensemble** | **~88.5%** | **~0.960** | **~43 ms** |

---

## 📁 Files in This Directory

```
deployment/
├── app.py                  # Flask API application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Docker Compose configuration
├── prometheus.yml         # Prometheus monitoring config
├── test_api.py            # API test suite
├── README.md              # This file
└── models/                # Model files (to be added)
    ├── random_forest_model.pkl
    ├── xgboost_model.json
    ├── deep_mlp_model.h5 (optional)
    ├── scaler.pkl
    ├── feature_names.pkl
    └── model_metadata.json
```

---

## 🎯 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional but recommended)
- Trained models from Google Colab

### Step 1: Export Models from Google Colab

Run this in your Colab notebook after training:

```python
import pickle
import xgboost as xgb
from tensorflow import keras
import json

# Create directory
!mkdir -p /content/drive/MyDrive/IDS_Research/deployment_models

# Save Random Forest
with open('/content/drive/MyDrive/IDS_Research/deployment_models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save XGBoost
xgb_model.save_model('/content/drive/MyDrive/IDS_Research/deployment_models/xgboost_model.json')

# Optional: Save MLP
# mlp_model.save('/content/drive/MyDrive/IDS_Research/deployment_models/deep_mlp_model.h5')

# Save scaler and feature names
with open('/content/drive/MyDrive/IDS_Research/deployment_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('/content/drive/MyDrive/IDS_Research/deployment_models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Save metadata
metadata = {
    'models': {
        'random_forest': {
            'accuracy': 0.877,
            'auc': 0.955,
            'inference_ms': 36.25,
            'file': 'random_forest_model.pkl'
        },
        'xgboost': {
            'accuracy': 0.876,
            'auc': 0.951,
            'inference_ms': 6.97,
            'file': 'xgboost_model.json'
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
        'features_count': len(feature_names),
        'feature_names_file': 'feature_names.pkl'
    },
    'version': '1.0.0',
    'created_at': '2025-11-18'
}

with open('/content/drive/MyDrive/IDS_Research/deployment_models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ All models saved!")
print("Download from: /content/drive/MyDrive/IDS_Research/deployment_models/")
```

### Step 2: Download Models

Download all files from Google Drive to your local `deployment/models/` directory:

```bash
# On your local machine
cd bhanuprasad-thesis/deployment
mkdir -p models

# Copy downloaded files to models/
cp ~/Downloads/random_forest_model.pkl models/
cp ~/Downloads/xgboost_model.json models/
cp ~/Downloads/scaler.pkl models/
cp ~/Downloads/feature_names.pkl models/
cp ~/Downloads/model_metadata.json models/
```

### Step 3: Test Locally (Docker)

```bash
# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f ids-api

# Test health check
curl http://localhost:5000/health

# Run test suite
python test_api.py
```

### Step 4: Test Locally (Without Docker)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

# In another terminal, test
curl http://localhost:5000/health
python test_api.py
```

---

## 🌐 AWS Deployment

### Option A: EC2 Instance (Recommended)

#### 1. Launch EC2 Instance

```bash
# AWS Console or CLI
Instance Type: t3.medium (2 vCPU, 4 GB RAM)
AMI: Ubuntu 22.04 LTS
Storage: 20 GB SSD
Security Group:
  - Port 22 (SSH) - Your IP
  - Port 5000 (API) - Your IP or 0.0.0.0/0
  - Port 9090 (Prometheus) - Your IP (optional)
  - Port 3000 (Grafana) - Your IP (optional)
```

#### 2. Connect to Instance

```bash
# Get instance public IP from AWS console
ssh -i your-key.pem ubuntu@your-ec2-ip
```

#### 3. Install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Logout and login again
exit
ssh -i your-key.pem ubuntu@your-ec2-ip

# Verify Docker
docker --version
```

#### 4. Upload Deployment Files

```bash
# From your local machine
scp -i your-key.pem -r deployment/ ubuntu@your-ec2-ip:~/

# Verify upload
ssh -i your-key.pem ubuntu@your-ec2-ip
cd deployment
ls -la models/  # Should show all .pkl and .json files
```

#### 5. Deploy with Docker Compose

```bash
# On EC2 instance
cd ~/deployment

# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f ids-api

# Test
curl http://localhost:5000/health
```

#### 6. Test from Your Machine

```bash
# From your local machine
curl http://your-ec2-ip:5000/health

# Or use test script
# Edit test_api.py: API_URL = "http://your-ec2-ip:5000"
python test_api.py
```

### Option B: AWS Lambda + API Gateway (Serverless)

For serverless deployment, see [AWS_LAMBDA_DEPLOYMENT.md](AWS_LAMBDA_DEPLOYMENT.md) (coming soon)

### Option C: AWS ECS/Fargate (Container Service)

For managed container deployment, see [AWS_ECS_DEPLOYMENT.md](AWS_ECS_DEPLOYMENT.md) (coming soon)

---

## 📡 API Endpoints

### GET /

Root endpoint with API information

```bash
curl http://localhost:5000/
```

### GET /health

Health check endpoint

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": {
    "random_forest": true,
    "xgboost": true,
    "deep_mlp": false
  },
  "features_count": 80,
  "version": "1.0.0"
}
```

### POST /predict

Single flow prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Dst Port": 80,
      "Protocol": 6,
      "Flow Duration": 120000,
      ...  # all 80 features required
    }
  }'
```

Response:
```json
{
  "prediction": 0,
  "prediction_label": "Benign",
  "confidence": 0.1523,
  "model_probabilities": {
    "random_forest": 0.1421,
    "xgboost": 0.1625
  },
  "inference_time_ms": 42.51,
  "timestamp": 1700000000.123
}
```

### POST /predict/batch

Batch prediction for multiple flows

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "flows": [
      {"Dst Port": 80, "Protocol": 6, ...},
      {"Dst Port": 443, "Protocol": 6, ...}
    ]
  }'
```

Response:
```json
{
  "predictions": [0, 1, 0],
  "probabilities": [0.15, 0.87, 0.23],
  "count": 3,
  "attack_count": 1,
  "benign_count": 2,
  "inference_time_ms": 125.43,
  "avg_time_per_flow_ms": 41.81
}
```

### GET /model/info

Model metadata and configuration

```bash
curl http://localhost:5000/model/info
```

### GET /metrics

Prometheus metrics (if enabled)

```bash
curl http://localhost:5000/metrics
```

---

## 📊 Monitoring

### Prometheus + Grafana (Optional)

Included in docker-compose.yml for monitoring:

```bash
# Access Prometheus
http://your-ec2-ip:9090

# Access Grafana
http://your-ec2-ip:3000
Username: admin
Password: admin

# Add Prometheus datasource in Grafana:
URL: http://prometheus:9090
```

### CloudWatch Monitoring

For AWS CloudWatch integration, install CloudWatch agent:

```bash
# On EC2 instance
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure agent to monitor:
# - CPU usage
# - Memory usage
# - Disk usage
# - Docker container metrics
# - Application logs
```

---

## 🔒 Security Best Practices

### 1. API Authentication

Add API key authentication to app.py:

```python
from functools import wraps
from flask import request

API_KEYS = {'your-secret-key'}  # Use environment variable in production

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key not in API_KEYS:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... existing code ...
```

### 2. HTTPS/SSL

Use AWS Certificate Manager + Application Load Balancer for HTTPS:

```bash
# Or use Let's Encrypt with nginx reverse proxy
sudo apt install nginx certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 3. Rate Limiting

Add rate limiting to prevent abuse:

```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... existing code ...
```

### 4. Firewall

Configure AWS Security Group or ufw:

```bash
# On EC2
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 5000/tcp  # API
sudo ufw enable
```

---

## 🐛 Troubleshooting

### Issue: Models not loading

```bash
# Check if model files exist
ls -lh models/

# Check file permissions
chmod 644 models/*

# Check Docker logs
docker-compose logs ids-api
```

### Issue: Out of memory

```bash
# Reduce number of workers in Dockerfile
CMD ["gunicorn", "--workers", "2", ...]  # Instead of 4

# Or upgrade EC2 instance type
# t3.medium (4 GB) → t3.large (8 GB)
```

### Issue: Slow inference

```bash
# Use XGBoost only (fastest)
# Edit metadata to use single model

# Or optimize batch size
# Process in smaller batches if memory is an issue
```

### Issue: Connection refused

```bash
# Check if container is running
docker ps

# Check if port is open
netstat -tlnp | grep 5000

# Check security group allows port 5000
# AWS Console → EC2 → Security Groups
```

---

## 📈 Performance Tuning

### Optimize Gunicorn Workers

```dockerfile
# Rule of thumb: (2 x CPU cores) + 1
# t3.medium (2 cores) → 5 workers

CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "5", \
     "--worker-class", "sync", \
     "--timeout", "120", \
     "app:app"]
```

### Enable Model Caching

Models are loaded once at startup (already implemented)

### Use Thread Pooling for Batch Requests

For very large batches, process in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

# Process batch in chunks
def process_batch_parallel(flows, chunk_size=1000):
    chunks = [flows[i:i+chunk_size] for i in range(0, len(flows), chunk_size)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(predict_chunk, chunks))
    return combine_results(results)
```

---

## 💰 Cost Estimation

### AWS EC2 Costs (Monthly)

| Instance Type | vCPU | RAM | Cost/Month | Recommended For |
|---------------|------|-----|------------|-----------------|
| t3.micro | 2 | 1 GB | FREE (first year) | Testing only |
| t3.small | 2 | 2 GB | $15.18 | Dev/staging |
| **t3.medium** | 2 | 4 GB | **$30.37** | **Production** |
| t3.large | 2 | 8 GB | $60.74 | High traffic |

### Additional Costs

- Data transfer: ~$0.09/GB (first 10 TB)
- EBS storage: $0.10/GB-month (20 GB = $2/month)
- Elastic IP: Free if attached, $3.65/month if not

### Total Estimated Cost

```
EC2 (t3.medium):     $30.37/month
EBS (20 GB):         $ 2.00/month
Data transfer (10GB): $ 0.90/month
--------------------------------
Total:               ~$35/month
```

---

## 🎓 Next Steps

1. **Export models from Colab** → Download to local
2. **Test locally** → Run with Docker
3. **Deploy to AWS EC2** → Launch instance, upload files
4. **Test deployment** → Run test_api.py
5. **Setup monitoring** → Prometheus + Grafana or CloudWatch
6. **Secure API** → Add authentication, HTTPS
7. **Optimize** → Tune workers, caching, batching
8. **Document** → API docs, runbooks for your team

---

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)
- [Docker Documentation](https://docs.docker.com/)
- [AWS EC2 User Guide](https://docs.aws.amazon.com/ec2/)
- [Prometheus Documentation](https://prometheus.io/docs/)

---

## ✅ Deployment Checklist

- [ ] Models trained and exported from Colab
- [ ] Models downloaded to local deployment/models/
- [ ] Tested locally with Docker
- [ ] Tested locally without Docker
- [ ] EC2 instance launched and configured
- [ ] Security groups configured
- [ ] Docker installed on EC2
- [ ] Files uploaded to EC2
- [ ] Container built and running
- [ ] Health check passes
- [ ] Prediction endpoint works
- [ ] Batch prediction works
- [ ] Monitoring setup (optional)
- [ ] Authentication added (recommended)
- [ ] HTTPS enabled (recommended)
- [ ] Documentation updated

---

**Your IDS API is ready for production!** 🚀

For questions or issues, refer to the [main project README](../README.md) or [PRODUCTION_MODEL_SELECTION.md](../PRODUCTION_MODEL_SELECTION.md).
