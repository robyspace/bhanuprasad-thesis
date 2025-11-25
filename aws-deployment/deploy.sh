#!/bin/bash
# IDS API Deployment Script
# Run this from your Mac to complete the deployment

set -e

EC2_IP="3.254.149.91"
KEY_FILE="ids-deploy-key.pem"
REMOTE_USER="ubuntu"

echo "=============================================="
echo "IDS API Deployment to AWS EC2"
echo "=============================================="
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "ERROR: SSH key file '$KEY_FILE' not found!"
    echo "Make sure you're running this script from the aws-deployment directory"
    exit 1
fi

# Set correct permissions on key file
chmod 400 $KEY_FILE

echo "Step 1: Testing SSH connection..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $KEY_FILE $REMOTE_USER@$EC2_IP "echo 'SSH connection successful!'"

echo ""
echo "Step 2: Waiting for Docker installation to complete..."
for i in {1..30}; do
    if ssh -i $KEY_FILE $REMOTE_USER@$EC2_IP "docker --version" 2>/dev/null; then
        echo "Docker is ready!"
        break
    fi
    echo "Waiting for Docker... (attempt $i/30)"
    sleep 10
done

echo ""
echo "Step 3: Creating deployment directory..."
ssh -i $KEY_FILE $REMOTE_USER@$EC2_IP "mkdir -p ~/ids-api/models ~/ids-api/logs"

echo ""
echo "Step 4: Uploading deployment files..."
scp -i $KEY_FILE ../deployment/app.py $REMOTE_USER@$EC2_IP:~/ids-api/
scp -i $KEY_FILE ../deployment/requirements.txt $REMOTE_USER@$EC2_IP:~/ids-api/
scp -i $KEY_FILE ../deployment/Dockerfile $REMOTE_USER@$EC2_IP:~/ids-api/
scp -i $KEY_FILE ../deployment/docker-compose.yml $REMOTE_USER@$EC2_IP:~/ids-api/
scp -i $KEY_FILE ../deployment/prometheus.yml $REMOTE_USER@$EC2_IP:~/ids-api/

echo ""
echo "Step 5: Uploading model files..."
scp -i $KEY_FILE ../models/scaler.pkl $REMOTE_USER@$EC2_IP:~/ids-api/models/
scp -i $KEY_FILE ../models/feature_names.pkl $REMOTE_USER@$EC2_IP:~/ids-api/models/
scp -i $KEY_FILE ../models/model_metadata.json $REMOTE_USER@$EC2_IP:~/ids-api/models/
scp -i $KEY_FILE ../models/xgboost_model.json $REMOTE_USER@$EC2_IP:~/ids-api/models/
scp -i $KEY_FILE ../models/deep_mlp_model.h5 $REMOTE_USER@$EC2_IP:~/ids-api/models/

echo ""
echo "=============================================="
echo "IMPORTANT: Upload your Random Forest model!"
echo "=============================================="
echo ""
echo "Run this command to upload your RF model:"
echo ""
echo "  scp -i $KEY_FILE /path/to/your/random_forest_model.pkl $REMOTE_USER@$EC2_IP:~/ids-api/models/"
echo ""
echo "Then start the application with:"
echo ""
echo "  ssh -i $KEY_FILE $REMOTE_USER@$EC2_IP 'cd ~/ids-api && docker compose up -d --build'"
echo ""
echo "=============================================="
