"""
IDS API - Network Intrusion Detection System
Ensemble of Random Forest + XGBoost models

Author: Your Name
Date: 2025-11
"""

from flask import Flask, request, jsonify, Response
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import json
import time
import logging
from logging.handlers import RotatingFileHandler
import os

# Prometheus metrics (optional - comment out if not using)
try:
    from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
    PROMETHEUS_ENABLED = True
    prediction_counter = Counter('predictions_total', 'Total predictions made', ['prediction_type'])
    prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
except ImportError:
    PROMETHEUS_ENABLED = False
    print("Warning: prometheus_client not installed. Metrics disabled.")

app = Flask(__name__)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

handler = RotatingFileHandler('logs/ids_api.log', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Load models and preprocessing
app.logger.info("Loading models...")

try:
    # Load metadata
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    app.logger.info("✓ Metadata loaded")

    # Load preprocessing
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    app.logger.info("✓ Scaler loaded")

    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    app.logger.info(f"✓ Feature names loaded ({len(feature_names)} features)")

    # Load Random Forest
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    app.logger.info("✓ Random Forest loaded")

    # Load XGBoost
    xgb_model = xgb.Booster()
    xgb_model.load_model('models/xgboost_model.json')
    app.logger.info("✓ XGBoost loaded")

    # Optional: Load MLP if exists
    try:
        from tensorflow import keras
        mlp_model = keras.models.load_model('models/deep_mlp_model.h5')
        app.logger.info("✓ Deep MLP loaded")
        USE_MLP = True
    except (FileNotFoundError, ImportError):
        app.logger.info("MLP model not found or TensorFlow not installed - using RF + XGB only")
        mlp_model = None
        USE_MLP = False

    app.logger.info("="*60)
    app.logger.info("ALL MODELS LOADED SUCCESSFULLY!")
    app.logger.info("="*60)

except Exception as e:
    app.logger.error(f"Error loading models: {str(e)}")
    raise


@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'IDS API - Network Intrusion Detection System',
        'version': '1.0.0',
        'status': 'running',
        'models': {
            'random_forest': True,
            'xgboost': True,
            'deep_mlp': USE_MLP
        },
        'endpoints': {
            'health': '/health',
            'single_prediction': '/predict (POST)',
            'batch_prediction': '/predict/batch (POST)',
            'model_info': '/model/info',
            'metrics': '/metrics (if Prometheus enabled)'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'random_forest': rf_model is not None,
            'xgboost': xgb_model is not None,
            'deep_mlp': USE_MLP
        },
        'features_count': len(feature_names),
        'version': '1.0.0',
        'timestamp': time.time()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict network flow as benign (0) or attack (1)

    Input JSON format:
    {
        "features": {
            "Dst Port": 80,
            "Protocol": 6,
            "Flow Duration": 1000,
            ... (all 80 features required)
        }
    }

    Response JSON format:
    {
        "prediction": 0 or 1,
        "prediction_label": "Benign" or "Attack",
        "confidence": 0.95,
        "model_probabilities": {
            "random_forest": 0.94,
            "xgboost": 0.96
        },
        "inference_time_ms": 42.5,
        "timestamp": 1699999999.999
    }
    """
    start_time = time.time()

    try:
        # Get input data
        data = request.get_json()

        if not data or 'features' not in data:
            app.logger.warning("Request missing 'features' field")
            return jsonify({'error': 'Missing features in request'}), 400

        # Convert to DataFrame
        features_dict = data['features']
        df = pd.DataFrame([features_dict])

        # Check for missing features
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            app.logger.warning(f"Missing features: {missing_features}")
            return jsonify({
                'error': f'Missing {len(missing_features)} features',
                'missing_features': list(missing_features)[:10]  # Show first 10
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

        # Get ensemble weights from metadata
        weights = metadata['deployment_config']['ensemble_weights']

        # Ensemble prediction (weighted average)
        if USE_MLP and mlp_model is not None:
            # Use all three models
            mlp_proba = mlp_model.predict(X_scaled, verbose=0)[0, 0]
            final_proba = (
                rf_proba * 0.35 +
                xgb_proba * 0.35 +
                mlp_proba * 0.30
            )
            model_probs = {
                'random_forest': float(rf_proba),
                'xgboost': float(xgb_proba),
                'deep_mlp': float(mlp_proba)
            }
        else:
            # Use RF + XGB only
            final_proba = (
                rf_proba * weights['random_forest'] +
                xgb_proba * weights['xgboost']
            )
            model_probs = {
                'random_forest': float(rf_proba),
                'xgboost': float(xgb_proba)
            }

        final_prediction = int(final_proba > 0.5)
        prediction_label = 'Attack' if final_prediction == 1 else 'Benign'

        inference_time = (time.time() - start_time) * 1000  # ms

        # Log prediction
        app.logger.info(
            f"Prediction: {prediction_label}, "
            f"confidence: {final_proba:.4f}, "
            f"time: {inference_time:.2f}ms"
        )

        # Update Prometheus metrics if enabled
        if PROMETHEUS_ENABLED:
            prediction_counter.labels(prediction_type=prediction_label).inc()

        # Return prediction
        return jsonify({
            'prediction': final_prediction,
            'prediction_label': prediction_label,
            'confidence': float(final_proba),
            'model_probabilities': model_probs,
            'inference_time_ms': round(inference_time, 2),
            'timestamp': time.time()
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple flows

    Input JSON format:
    {
        "flows": [
            {"Dst Port": 80, "Protocol": 6, ...},
            {"Dst Port": 443, "Protocol": 6, ...},
            ...
        ]
    }

    Response JSON format:
    {
        "predictions": [0, 1, 0, ...],
        "probabilities": [0.15, 0.87, 0.23, ...],
        "count": 100,
        "attack_count": 23,
        "benign_count": 77,
        "inference_time_ms": 1234.56,
        "avg_time_per_flow_ms": 12.35
    }
    """
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or 'flows' not in data:
            app.logger.warning("Batch request missing 'flows' field")
            return jsonify({'error': 'Missing flows in request'}), 400

        flows = data['flows']

        if len(flows) == 0:
            return jsonify({'error': 'Empty flows array'}), 400

        if len(flows) > 10000:
            app.logger.warning(f"Large batch request: {len(flows)} flows")
            return jsonify({'error': 'Batch size too large (max 10,000)'}), 400

        df = pd.DataFrame(flows)

        # Ensure all features present
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing {len(missing_features)} features',
                'missing_features': list(missing_features)[:10]
            }), 400

        df = df[feature_names]

        # Scale
        X_scaled = scaler.transform(df)

        # Random Forest predictions
        rf_proba = rf_model.predict_proba(X_scaled)[:, 1]

        # XGBoost predictions
        dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
        xgb_proba = xgb_model.predict(dmatrix)

        # Ensemble
        weights = metadata['deployment_config']['ensemble_weights']

        if USE_MLP and mlp_model is not None:
            mlp_proba = mlp_model.predict(X_scaled, verbose=0).flatten()
            final_proba = (
                rf_proba * 0.35 +
                xgb_proba * 0.35 +
                mlp_proba * 0.30
            )
        else:
            final_proba = (
                rf_proba * weights['random_forest'] +
                xgb_proba * weights['xgboost']
            )

        predictions = (final_proba > 0.5).astype(int).tolist()

        inference_time = (time.time() - start_time) * 1000

        attack_count = int(sum(predictions))
        benign_count = len(predictions) - attack_count

        app.logger.info(
            f"Batch prediction: {len(predictions)} flows, "
            f"{attack_count} attacks, {benign_count} benign, "
            f"time: {inference_time:.2f}ms"
        )

        # Update Prometheus metrics if enabled
        if PROMETHEUS_ENABLED:
            prediction_counter.labels(prediction_type='Attack').inc(attack_count)
            prediction_counter.labels(prediction_type='Benign').inc(benign_count)

        return jsonify({
            'predictions': predictions,
            'probabilities': final_proba.tolist(),
            'count': len(predictions),
            'attack_count': attack_count,
            'benign_count': benign_count,
            'inference_time_ms': round(inference_time, 2),
            'avg_time_per_flow_ms': round(inference_time / len(predictions), 2)
        })

    except Exception as e:
        app.logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Return model metadata and performance metrics"""
    return jsonify(metadata)


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    if not PROMETHEUS_ENABLED:
        return jsonify({'error': 'Prometheus metrics not enabled'}), 404

    return Response(generate_latest(REGISTRY), mimetype='text/plain')


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("="*60)
    print("IDS API Server")
    print("="*60)
    print("Models loaded and ready!")
    print("Access API at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("API docs: http://localhost:5000/")
    print("="*60)

    # Run with Flask development server (use gunicorn in production)
    app.run(host='0.0.0.0', port=5000, debug=False)
