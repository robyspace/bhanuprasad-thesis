## 📋 Overview

This guide takes you from model training in Google Colab to running live inference on AWS EC2 with Prometheus monitoring.

**Timeline:** 3-5 days **Cost:** ~$35/month for AWS (can use free tier for testing)

---

## Part 1: Model Training on Google Colab (Days 1-2)

### Step 1.1: Environment Setup (15 minutes)

```python
# Open new Google Colab notebook
# https://colab.research.google.com

# Install required libraries
!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn kaggle

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create project directory
import os
project_dir = '/content/drive/MyDrive/IDS_Research'
os.makedirs(project_dir, exist_ok=True)
os.makedirs(f'{project_dir}/models', exist_ok=True)
os.makedirs(f'{project_dir}/results', exist_ok=True)
os.makedirs(f'{project_dir}/deployment', exist_ok=True)

print(f"✓ Project directory created at: {project_dir}")
```

### Step 1.2: Download Dataset (30 minutes)

```python
# Setup Kaggle API
from google.colab import files
import os

# Upload your kaggle.json file when prompted
print("Please upload your kaggle.json file:")
uploaded = files.upload()

# Configure Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download CSE-CIC-IDS-2018 dataset
!mkdir -p /content/data
!kaggle datasets download -d dhoogla/csecicids2018 -p /content/data --unzip

print("✓ Dataset downloaded successfully")
```

### Step 1.3: Data Preprocessing (1-2 hours)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Load parquet files
data_dir = '/content/data'
parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

print(f"Found {len(parquet_files)} parquet files")

# Load first file for quick start (or load all for full dataset)
df = pd.read_parquet(os.path.join(data_dir, parquet_files[0]))

# Identify label column
label_col = 'Label' if 'Label' in df.columns else df.columns[-1]

# Separate features and labels
X = df.drop(columns=[label_col])
y = df[label_col]

# Create binary labels (Benign = 0, Attack = 1)
y_binary = (y != 'Benign').astype(int)

# Remove duplicates and handle missing values
X = X.drop_duplicates()
y_binary = y_binary[X.index]

# Handle infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Remove constant columns
constant_cols = [col for col in X_imputed.columns if X_imputed[col].nunique() <= 1]
X_clean = X_imputed.drop(columns=constant_cols)

print(f"✓ Preprocessed data shape: {X_clean.shape}")
print(f"✓ Label distribution: Benign={sum(y_binary==0)}, Attack={sum(y_binary==1)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"✓ Training set: {X_train_scaled.shape[0]} samples")
print(f"✓ Test set: {X_test_scaled.shape[0]} samples")
```

### Step 1.4: Train XGBoost Model (30 minutes)

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# Calculate class weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost
print("Training XGBoost model...")
start_time = time.time()

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"✓ Training completed in {training_time:.2f} seconds")

# Evaluate
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✓ XGBoost Performance:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print(f"  ROC-AUC:   {roc_auc:.4f}")
```

### Step 1.5: Train Random Forest Model (30 minutes)

```python
from sklearn.ensemble import RandomForestClassifier

print("Training Random Forest model...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
training_time_rf = time.time() - start_time

print(f"✓ Training completed in {training_time_rf:.2f} seconds")

# Evaluate
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\n✓ Random Forest Performance:")
print(f"  Accuracy:  {accuracy_rf*100:.2f}%")
print(f"  Precision: {precision_rf*100:.2f}%")
print(f"  Recall:    {recall_rf*100:.2f}%")
print(f"  F1-Score:  {f1_rf*100:.2f}%")
print(f"  ROC-AUC:   {roc_auc_rf:.4f}")
```

### Step 1.6: Save Models for Deployment (15 minutes)

```python
import pickle
import json

# Save models
deployment_dir = f'{project_dir}/deployment'

print("Saving models for deployment...")

# Save XGBoost model
with open(f'{deployment_dir}/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Save Random Forest model
with open(f'{deployment_dir}/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save scaler
with open(f'{deployment_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open(f'{deployment_dir}/feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Save model metadata
metadata = {
    'xgboost': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'feature_count': len(X_train.columns)
    },
    'random_forest': {
        'accuracy': float(accuracy_rf),
        'precision': float(precision_rf),
        'recall': float(recall_rf),
        'f1_score': float(f1_rf),
        'roc_auc': float(roc_auc_rf),
        'feature_count': len(X_train.columns)
    },
    'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
}

with open(f'{deployment_dir}/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✓ All models saved successfully!")
print(f"\nSaved files in: {deployment_dir}")
print("  - xgboost_model.pkl")
print("  - random_forest_model.pkl")
print("  - scaler.pkl")
print("  - feature_names.pkl")
print("  - model_metadata.json")
```

### Step 1.7: Download Deployment Package (10 minutes)

```python
import shutil
from google.colab import files

# Create deployment package
package_dir = '/content/deployment_package'
os.makedirs(package_dir, exist_ok=True)

# Copy model files
shutil.copy(f'{deployment_dir}/xgboost_model.pkl', package_dir)
shutil.copy(f'{deployment_dir}/random_forest_model.pkl', package_dir)
shutil.copy(f'{deployment_dir}/scaler.pkl', package_dir)
shutil.copy(f'{deployment_dir}/feature_names.pkl', package_dir)
shutil.copy(f'{deployment_dir}/model_metadata.json', package_dir)

# Create ZIP file
shutil.make_archive('deployment_package', 'zip', package_dir)

# Download
print("Downloading deployment package...")
files.download('deployment_package.zip')

print("✓ Download complete!")
print("\nNext: Extract this ZIP file and move to Part 2 (AWS Deployment)")
```

---

## Part 2: AWS Setup (Day 3)

### Step 2.1: Create AWS Account & Launch EC2 Instance (30 minutes)

**Via AWS Console:**

1. **Go to AWS Console**: https://console.aws.amazon.com/
2. **Navigate to EC2**: Services → EC2
3. **Launch Instance**:
    - Click "Launch Instance"
    - **Name**: `IDS-API-Server`
    - **AMI**: Ubuntu Server 20.04 LTS (Free tier eligible)
    - **Instance Type**: `t2.medium` (2 vCPU, 4 GB RAM) _or t2.micro for free tier_
    - **Key Pair**: Create new → Download `.pem` file → Save securely
    - **Network Settings**:
        - Create security group: `ids-api-sg`
        - Allow SSH (port 22) from My IP
        - Allow Custom TCP (port 5000) from Anywhere
        - Allow Custom TCP (port 9090) from Anywhere
        - Allow Custom TCP (port 3000) from Anywhere
    - **Storage**: 20 GB gp3
4. **Launch Instance**
5. **Note the Public IPv4 address** (e.g., `3.85.123.45`)

### Step 2.2: Connect to EC2 Instance (10 minutes)

```bash
# On your local machine (Mac/Linux)
# Replace with your key and EC2 IP
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@3.85.123.45

# On Windows (use PuTTY or Git Bash)
# If using Git Bash:
ssh -i your-key.pem ubuntu@3.85.123.45
```

### Step 2.3: Install Docker on EC2 (15 minutes)

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version

echo "✓ Docker installed successfully"
echo "Please log out and log back in for group changes to take effect"
```

```bash
# Log out
exit

# Log back in
ssh -i your-key.pem ubuntu@3.85.123.45
```

---

## Part 3: Deployment Files Creation (Day 3 continued)

### Step 3.1: Create Flask API Application

```bash
# Create deployment directory
mkdir -p ~/ids-deployment
cd ~/ids-deployment
```

Create `app.py`:

```bash
nano app.py
```

Paste this content:

```python
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import psutil

app = Flask(__name__)

# Load models
print("Loading models...")
with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
print("✓ Models loaded")

# Prometheus metrics
predictions_total = Counter('ids_predictions_total', 'Total predictions', ['model', 'prediction'])
prediction_latency = Histogram('ids_prediction_latency_seconds', 'Prediction latency', ['model'])
cpu_usage = Gauge('ids_cpu_usage_percent', 'CPU usage')
memory_usage = Gauge('ids_memory_usage_percent', 'Memory usage')

@app.route('/health', methods=['GET'])
def health():
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)
    return jsonify({
        'status': 'healthy',
        'models': ['xgboost', 'random_forest'],
        'cpu': psutil.cpu_percent(),
        'memory': psutil.virtual_memory().percent
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_name = data.get('model', 'xgboost').lower()
        features = data.get('features')
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Select model
        model = xgb_model if model_name == 'xgboost' else rf_model
        
        # Prepare features
        df = pd.DataFrame([features])
        df = df[feature_names]
        X_scaled = scaler.transform(df)
        
        # Predict with timing
        start = time.time()
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        latency = time.time() - start
        
        # Update metrics
        prediction_latency.labels(model=model_name).observe(latency)
        pred_label = 'attack' if prediction == 1 else 'benign'
        predictions_total.labels(model=model_name, prediction=pred_label).inc()
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().percent)
        
        return jsonify({
            'model': model_name,
            'prediction': int(prediction),
            'prediction_label': pred_label,
            'confidence': float(proba[prediction]),
            'probabilities': {'benign': float(proba[0]), 'attack': float(proba[1])},
            'latency_ms': latency * 1000
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Step 3.2: Create Requirements File

```bash
nano requirements.txt
```

Paste:

```
flask==2.3.0
xgboost==1.7.6
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
prometheus-client==0.17.1
psutil==5.9.5
gunicorn==21.2.0
```

Save and exit.

### Step 3.3: Create Dockerfile

```bash
nano Dockerfile
```

Paste:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY *.pkl .
COPY model_metadata.json .

EXPOSE 5000

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

Save and exit.

### Step 3.4: Create Prometheus Configuration

```bash
nano prometheus.yml
```

Paste:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ids-api'
    static_configs:
      - targets: ['ids-api:5000']
```

Save and exit.

### Step 3.5: Create Docker Compose File

```bash
nano docker-compose.yml
```

Paste:

```yaml
version: '3.8'

services:
  ids-api:
    build: .
    ports:
      - "5000:5000"
    restart: unless-stopped
    networks:
      - monitoring

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    networks:
      - monitoring
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - monitoring
    restart: unless-stopped

networks:
  monitoring:

volumes:
  prometheus-data:
  grafana-data:
```

Save and exit.

### Step 3.6: Upload Model Files to EC2

**On your local machine** (new terminal):

```bash
# Extract the deployment_package.zip you downloaded from Colab
unzip deployment_package.zip

# Upload files to EC2
scp -i your-key.pem deployment_package/* ubuntu@3.85.123.45:~/ids-deployment/

# Verify upload
ssh -i your-key.pem ubuntu@3.85.123.45 "ls -lh ~/ids-deployment/"
```

You should see:

- xgboost_model.pkl
- random_forest_model.pkl
- scaler.pkl
- feature_names.pkl
- model_metadata.json
- app.py
- Dockerfile
- docker-compose.yml
- prometheus.yml
- requirements.txt

---

## Part 4: Deploy and Test (Day 4)

### Step 4.1: Build and Start Services

```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@3.85.123.45

# Navigate to deployment directory
cd ~/ids-deployment

# Build Docker image
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

You should see 3 services running:

- ids-api
- prometheus
- grafana

### Step 4.2: Verify API is Running

```bash
# Test health endpoint
curl http://localhost:5000/health

# Expected output:
# {"status": "healthy", "models": ["xgboost", "random_forest"], "cpu": 5.2, "memory": 45.3}
```

### Step 4.3: Test Prediction (on EC2)

```bash
# Create test data (adjust features based on your model)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xgboost",
    "features": {
      "feature_0": 0.5,
      "feature_1": 1.2,
      "feature_2": -0.3
    }
  }'

# Expected output:
# {
#   "model": "xgboost",
#   "prediction": 0,
#   "prediction_label": "benign",
#   "confidence": 0.98,
#   "probabilities": {"benign": 0.98, "attack": 0.02},
#   "latency_ms": 5.23
# }
```

### Step 4.4: Test from External Machine

**On your local machine**:

```bash
# Replace with your EC2 public IP
curl http://3.85.123.45:5000/health

# Test prediction
curl -X POST http://3.85.123.45:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xgboost",
    "features": {
      "feature_0": 0.5,
      "feature_1": 1.2,
      "feature_2": -0.3
    }
  }'
```

### Step 4.5: Access Monitoring Dashboards

**Prometheus**:

- URL: `http://3.85.123.45:9090`
- Query examples:
    - `rate(ids_predictions_total[1m])`
    - `histogram_quantile(0.95, rate(ids_prediction_latency_seconds_bucket[5m]))`
    - `ids_cpu_usage_percent`

**Grafana**:

- URL: `http://3.85.123.45:3000`
- Login: `admin` / `admin`
- Add Prometheus datasource: `http://prometheus:9090`

---

## Part 5: Performance Testing (Day 5)

### Step 5.1: Create Test Script

**On your local machine**, create `test_aws_api.py`:

```python
import requests
import time
import statistics
import json

# Configuration
API_URL = "http://3.85.123.45:5000"  # Replace with your EC2 IP
NUM_REQUESTS = 100

# Generate sample features (adjust based on your model)
def get_sample_features():
    import random
    return {f'feature_{i}': random.uniform(-2, 2) for i in range(10)}

# Run test
results = {'latencies': [], 'successes': 0, 'errors': 0}

print(f"Testing API at {API_URL}")
print(f"Running {NUM_REQUESTS} requests...")

start_time = time.time()

for i in range(NUM_REQUESTS):
    try:
        features = get_sample_features()
        req_start = time.time()
        
        response = requests.post(
            f"{API_URL}/predict",
            json={'model': 'xgboost', 'features': features},
            timeout=10
        )
        
        latency = (time.time() - req_start) * 1000
        
        if response.status_code == 200:
            results['successes'] += 1
            results['latencies'].append(latency)
        else:
            results['errors'] += 1
            
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{NUM_REQUESTS}")
            
    except Exception as e:
        results['errors'] += 1
        print(f"Error: {e}")

duration = time.time() - start_time

# Print results
print(f"\n{'='*60}")
print("TEST RESULTS")
print(f"{'='*60}")
print(f"Duration: {duration:.2f}s")
print(f"Successful: {results['successes']}")
print(f"Failed: {results['errors']}")
print(f"Throughput: {NUM_REQUESTS/duration:.2f} req/s")
print(f"\nLatency:")
print(f"  Mean: {statistics.mean(results['latencies']):.2f} ms")
print(f"  Median: {statistics.median(results['latencies']):.2f} ms")
print(f"  Min: {min(results['latencies']):.2f} ms")
print(f"  Max: {max(results['latencies']):.2f} ms")
print(f"{'='*60}")
```

### Step 5.2: Run Tests

```bash
python test_aws_api.py
```

Expected output:

```
Testing API at http://3.85.123.45:5000
Running 100 requests...
Progress: 10/100
Progress: 20/100
...
============================================================
TEST RESULTS
============================================================
Duration: 12.45s
Successful: 100
Failed: 0
Throughput: 8.03 req/s

Latency:
  Mean: 8.23 ms
  Median: 7.85 ms
  Min: 5.12 ms
  Max: 15.67 ms
============================================================
```

---

## Part 6: Collecting Metrics for Research Paper

### Step 6.1: Query Prometheus Metrics

```python
import requests
import pandas as pd

PROMETHEUS_URL = "http://3.85.123.45:9090"

# Query prediction rate
query = "rate(ids_predictions_total[1m])"
response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
data = response.json()
print("Prediction Rate:", data['data']['result'])

# Query P95 latency
query = "histogram_quantile(0.95, rate(ids_prediction_latency_seconds_bucket[5m]))"
response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
data = response.json()
p95_latency = float(data['data']['result'][0]['value'][1]) * 1000
print(f"P95 Latency: {p95_latency:.2f} ms")

# Query CPU usage
query = "ids_cpu_usage_percent"
response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
data = response.json()
cpu = float(data['data']['result'][0]['value'][1])
print(f"CPU Usage: {cpu:.2f}%")
```

### Step 6.2: Generate Performance Report

Create `generate_report.py`:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load model metadata
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Create performance comparison table
df = pd.DataFrame({
    'Model': ['XGBoost', 'Random Forest'],
    'Accuracy': [metadata['xgboost']['accuracy'], metadata['random_forest']['accuracy']],
    'Precision': [metadata['xgboost']['precision'], metadata['random_forest']['precision']],
    'Recall': [metadata['xgboost']['recall'], metadata['random_forest']['recall']],
    'F1-Score': [metadata['xgboost']['f1_score'], metadata['random_forest']['f1_score']],
    'ROC-AUC': [metadata['xgboost']['roc_auc'], metadata['random_forest']['roc_auc']]
})

print("\nModel Performance Comparison:")
print(df.to_string(index=False))

# Save for paper
df.to_csv('model_comparison.csv', index=False)
df.to_latex('model_comparison.tex', index=False, float_format="%.4f")

print("\n✓ Report saved as model_comparison.csv and model_comparison.tex")
```

---

## 🎯 Summary Checklist

### ✅ Part 1: Colab Training (Completed when you have)

- [ ] Models trained (XGBoost, Random Forest)
- [ ] Models saved as .pkl files
- [ ] Scaler and feature names saved
- [ ] Metadata JSON created
- [ ] deployment_package.zip downloaded

### ✅ Part 2: AWS Setup (Completed when you have)

- [ ] EC2 instance launched
- [ ] Security groups configured
- [ ] SSH access working
- [ ] Docker installed

### ✅ Part 3: Deployment (Completed when you have)

- [ ] app.py created
- [ ] Dockerfile created
- [ ] docker-compose.yml created
- [ ] Model files uploaded to EC2
- [ ] All services running

### ✅ Part 4: Testing (Completed when you have)

- [ ] Health endpoint working
- [ ] Predictions working locally on EC2
- [ ] Predictions working from external machine
- [ ] Prometheus collecting metrics
- [ ] Grafana accessible

### ✅ Part 5: Performance Testing (Completed when you have)

- [ ] Load tests run successfully
- [ ] Latency metrics collected
- [ ] Throughput measured
- [ ] CPU/Memory usage monitored

### ✅ Part 6: Research Metrics (Completed when you have)

- [ ] Model performance metrics documented
- [ ] Deployment metrics collected
- [ ] Comparison tables generated
- [ ] Figures/plots created for paper

---

## 🚨 Troubleshooting

### Issue: "Permission denied" when connecting to EC2

```bash
chmod 400 your-key.pem
```

### Issue: "Connection refused" on port 5000

```bash
# Check if service is running
docker-compose ps

# View logs
docker-compose logs ids-api

# Restart service
docker-compose restart ids-api
```

### Issue: Models not loading

```bash
# Verify files exist
ls -lh ~/ids-deployment/*.pkl

# Check file sizes (should not be 0 bytes)
# If files are corrupt, re-upload from local machine
```

### Issue: Out of memory

```bash
# Check memory usage
docker stats

# Solution: Upgrade to larger instance (t2.medium → t2.large)
```

---

## 💰 Cost Breakdown

- **t2.medium**: $0.0464/hour × 730 hours = $33.87/month
- **t2.micro (free tier)**: $0 for first 750 hours/month
- **Storage (20GB)**: ~$1.60/month
- **Data transfer**: ~$1/month

**Total: ~$35-40/month** (or free with t2.micro during testing)

---

## 📚 Next Steps

1. **Collect more metrics** - Run tests for 24 hours
2. **Create visualizations** - Generate plots for paper
3. **Write paper sections** - Use collected metrics
4. **Optimize if needed** - Tune model parameters
5. **Clean up** - Stop instances when not needed

**Your AWS deployment is now complete and ready for research! 🎉**