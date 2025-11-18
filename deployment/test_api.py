"""
Test script for IDS API
Tests all endpoints with sample data
"""

import requests
import json
import time
import sys

# Configuration
API_URL = "http://localhost:5000"  # Change to your EC2 IP when deployed
SAMPLE_FEATURES_FILE = "sample_flow.json"  # Optional: load from file


def print_separator():
    print("\n" + "="*60 + "\n")


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print("✓ Health check passed!")
        print(f"  Status: {data['status']}")
        print(f"  Models loaded: {data['models_loaded']}")
        print(f"  Features: {data['features_count']}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return False


def test_home():
    """Test root endpoint"""
    print("Testing / endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        response.raise_for_status()
        data = response.json()
        print("✓ Root endpoint passed!")
        print(f"  Service: {data['service']}")
        print(f"  Version: {data['version']}")
        return True
    except Exception as e:
        print(f"✗ Root endpoint failed: {str(e)}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("Testing /model/info endpoint...")
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        response.raise_for_status()
        data = response.json()
        print("✓ Model info retrieved!")
        print(f"  Models: {list(data['models'].keys())}")
        print(f"  Primary model: {data['deployment_config']['primary_model']}")
        return True
    except Exception as e:
        print(f"✗ Model info failed: {str(e)}")
        return False


def test_single_prediction():
    """Test single prediction endpoint"""
    print("Testing /predict endpoint...")

    # Sample flow features (you'll need to fill these with actual values)
    # This is a minimal example - you need all 80 features
    sample_flow = {
        "features": {
            "Dst Port": 80,
            "Protocol": 6,
            "Flow Duration": 120000,
            "Tot Fwd Pkts": 10,
            "Tot Bwd Pkts": 8,
            "TotLen Fwd Pkts": 1500,
            "TotLen Bwd Pkts": 1200,
            # ... add all other features here ...
            # You can get these from your test set
        }
    }

    try:
        start = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=sample_flow,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 400:
            error_data = response.json()
            print(f"⚠ Prediction requires all features")
            print(f"  Error: {error_data.get('error')}")
            if 'missing_features' in error_data:
                print(f"  Missing: {error_data['missing_features'][:5]}...")
            return False

        response.raise_for_status()
        elapsed = (time.time() - start) * 1000

        data = response.json()
        print("✓ Single prediction passed!")
        print(f"  Prediction: {data['prediction_label']}")
        print(f"  Confidence: {data['confidence']:.4f}")
        print(f"  Model probabilities: {data['model_probabilities']}")
        print(f"  Inference time: {data['inference_time_ms']:.2f} ms")
        print(f"  Total roundtrip: {elapsed:.2f} ms")
        return True

    except Exception as e:
        print(f"✗ Single prediction failed: {str(e)}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("Testing /predict/batch endpoint...")

    # Create 10 identical sample flows for testing
    sample_flow_features = {
        "Dst Port": 443,
        "Protocol": 6,
        "Flow Duration": 60000,
        # ... add all features ...
    }

    batch_data = {
        "flows": [sample_flow_features] * 10
    }

    try:
        start = time.time()
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        if response.status_code == 400:
            error_data = response.json()
            print(f"⚠ Batch prediction requires all features")
            print(f"  Error: {error_data.get('error')}")
            return False

        response.raise_for_status()
        elapsed = (time.time() - start) * 1000

        data = response.json()
        print("✓ Batch prediction passed!")
        print(f"  Total flows: {data['count']}")
        print(f"  Attacks detected: {data['attack_count']}")
        print(f"  Benign flows: {data['benign_count']}")
        print(f"  Total inference time: {data['inference_time_ms']:.2f} ms")
        print(f"  Avg per flow: {data['avg_time_per_flow_ms']:.2f} ms")
        print(f"  Total roundtrip: {elapsed:.2f} ms")
        return True

    except Exception as e:
        print(f"✗ Batch prediction failed: {str(e)}")
        return False


def test_metrics():
    """Test Prometheus metrics endpoint"""
    print("Testing /metrics endpoint...")
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)

        if response.status_code == 404:
            print("⚠ Metrics endpoint not available (Prometheus not enabled)")
            return True

        response.raise_for_status()
        print("✓ Metrics endpoint passed!")
        print(f"  Metrics available: {len(response.text.split('\\n'))} lines")
        return True

    except Exception as e:
        print(f"✗ Metrics failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("IDS API Test Suite")
    print("="*60)
    print(f"Testing API at: {API_URL}")
    print_separator()

    tests = [
        ("Health Check", test_health),
        ("Home Endpoint", test_home),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Metrics", test_metrics),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"Test: {test_name}")
        result = test_func()
        results.append((test_name, result))
        print_separator()
        time.sleep(0.5)  # Small delay between tests

    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
