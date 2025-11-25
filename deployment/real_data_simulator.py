#!/usr/bin/env python3
"""
IDS Real Data Simulator - End-to-End Testing with Actual CSE-CIC-IDS-2018 Data
Uses real test data from your training pipeline for accurate model validation.

Usage:
    python real_data_simulator.py --samples 100 --api http://3.254.149.91:5000
    python real_data_simulator.py --samples 500 --mode attack --verbose
"""

import requests
import json
import time
import random
import argparse
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import sys

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy required. Install with: pip install pandas numpy pyarrow")
    sys.exit(1)

# Default paths
DATA_DIR = "processed_data"
API_URL = "http://3.254.149.91:5000"

# Statistics
stats = {
    "total_requests": 0,
    "benign_predicted": 0,
    "attack_predicted": 0,
    "true_positives": 0,
    "false_positives": 0,
    "true_negatives": 0,
    "false_negatives": 0,
    "errors": 0,
    "total_latency_ms": 0,
}
stats_lock = threading.Lock()


def load_test_data(data_dir):
    """Load test data from parquet files."""
    print(f"Loading test data from {data_dir}...")

    X_test = pd.read_parquet(f"{data_dir}/X_test_scaled.parquet")
    y_test = pd.read_parquet(f"{data_dir}/y_test.parquet")

    with open(f"{data_dir}/feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)

    with open(f"{data_dir}/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)

    print(f"  Loaded {len(X_test)} samples")
    print(f"  Features: {len(feature_names)}")
    print(f"  Benign: {(y_test['Label'] == 0).sum()}, Attack: {(y_test['Label'] == 1).sum()}")

    return X_test, y_test, feature_names, scaler


def inverse_scale(X_scaled, scaler):
    """Convert scaled data back to original feature values."""
    return pd.DataFrame(
        scaler.inverse_transform(X_scaled),
        columns=X_scaled.columns,
        index=X_scaled.index
    )


def send_prediction(features_dict, actual_label, api_url, sample_idx):
    """Send a prediction request to the API."""
    global stats

    try:
        start_time = time.time()
        response = requests.post(
            f"{api_url}/predict",
            json={"features": features_dict},
            timeout=30
        )
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            predicted_label = 1 if result["prediction"] == 1 else 0

            with stats_lock:
                stats["total_requests"] += 1
                stats["total_latency_ms"] += latency

                if predicted_label == 1:
                    stats["attack_predicted"] += 1
                else:
                    stats["benign_predicted"] += 1

                # Confusion matrix
                if actual_label == 1 and predicted_label == 1:
                    stats["true_positives"] += 1
                elif actual_label == 0 and predicted_label == 1:
                    stats["false_positives"] += 1
                elif actual_label == 0 and predicted_label == 0:
                    stats["true_negatives"] += 1
                elif actual_label == 1 and predicted_label == 0:
                    stats["false_negatives"] += 1

            return {
                "success": True,
                "sample_idx": sample_idx,
                "actual": actual_label,
                "predicted": predicted_label,
                "confidence": result["confidence"],
                "latency_ms": latency,
                "correct": actual_label == predicted_label
            }
        else:
            with stats_lock:
                stats["errors"] += 1
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:100]}"}

    except Exception as e:
        with stats_lock:
            stats["errors"] += 1
        return {"success": False, "error": str(e)}


def run_simulation(data_dir, api_url, num_samples, mode, rate, verbose, batch_size):
    """Run the simulation with real test data."""
    global stats

    # Reset stats
    for key in stats:
        stats[key] = 0

    # Load data
    X_test, y_test, feature_names, scaler = load_test_data(data_dir)

    # Inverse scale to get original feature values
    print("\nInverse scaling data to original feature values...")
    X_original = inverse_scale(X_test, scaler)

    # Filter by mode
    if mode == "benign":
        mask = y_test['Label'] == 0
        X_filtered = X_original[mask]
        y_filtered = y_test[mask]
        print(f"Filtered to {len(X_filtered)} benign samples")
    elif mode == "attack":
        mask = y_test['Label'] == 1
        X_filtered = X_original[mask]
        y_filtered = y_test[mask]
        print(f"Filtered to {len(X_filtered)} attack samples")
    else:  # mixed
        X_filtered = X_original
        y_filtered = y_test

    # Sample data
    if num_samples > len(X_filtered):
        num_samples = len(X_filtered)
        print(f"Adjusted to {num_samples} samples (max available)")

    indices = random.sample(range(len(X_filtered)), num_samples)

    print(f"\n{'='*60}")
    print(f"IDS Real Data Simulator")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Samples: {num_samples}")
    print(f"Rate: {rate} requests/second")
    print(f"API: {api_url}")
    print(f"{'='*60}\n")

    # Check if using batch mode
    if batch_size and batch_size > 1:
        run_batch_simulation(X_filtered, y_filtered, feature_names, indices, api_url, batch_size, verbose)
    else:
        run_single_simulation(X_filtered, y_filtered, feature_names, indices, api_url, rate, verbose)

    print_statistics()


def run_single_simulation(X_filtered, y_filtered, feature_names, indices, api_url, rate, verbose):
    """Run simulation sending one request at a time."""
    interval = 1.0 / rate if rate > 0 else 0

    print("Starting single-request simulation...")
    print("Press Ctrl+C to stop early\n")

    try:
        for i, idx in enumerate(indices):
            # Get sample
            features = X_filtered.iloc[idx].to_dict()
            actual_label = int(y_filtered.iloc[idx]['Label'])

            # Send prediction
            result = send_prediction(features, actual_label, api_url, idx)

            if verbose and result["success"]:
                actual_str = "Attack" if result["actual"] == 1 else "Benign"
                pred_str = "Attack" if result["predicted"] == 1 else "Benign"
                status = "✓" if result["correct"] else "✗"
                print(f"{status} Sample {idx:6d} | Actual: {actual_str:6s} | "
                      f"Predicted: {pred_str:6s} | Confidence: {result['confidence']:.3f} | "
                      f"Latency: {result['latency_ms']:.1f}ms")
            elif not verbose:
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"\rProgress: {i+1}/{len(indices)} samples | "
                          f"Accuracy: {(stats['true_positives'] + stats['true_negatives']) / max(stats['total_requests'], 1) * 100:.1f}%",
                          end="")

            if interval > 0:
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")


def run_batch_simulation(X_filtered, y_filtered, feature_names, indices, api_url, batch_size, verbose):
    """Run simulation using batch endpoint."""
    print(f"Starting batch simulation (batch size: {batch_size})...")
    print("Press Ctrl+C to stop early\n")

    global stats

    try:
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]

            # Prepare batch
            flows = []
            actual_labels = []
            for idx in batch_indices:
                features = X_filtered.iloc[idx].to_dict()
                flows.append(features)
                actual_labels.append(int(y_filtered.iloc[idx]['Label']))

            # Send batch request
            start_time = time.time()
            try:
                response = requests.post(
                    f"{api_url}/predict/batch",
                    json={"flows": flows},
                    timeout=60
                )
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    result = response.json()
                    predictions = result["predictions"]

                    with stats_lock:
                        stats["total_requests"] += len(predictions)
                        stats["total_latency_ms"] += latency

                        for actual, predicted in zip(actual_labels, predictions):
                            if predicted == 1:
                                stats["attack_predicted"] += 1
                            else:
                                stats["benign_predicted"] += 1

                            if actual == 1 and predicted == 1:
                                stats["true_positives"] += 1
                            elif actual == 0 and predicted == 1:
                                stats["false_positives"] += 1
                            elif actual == 0 and predicted == 0:
                                stats["true_negatives"] += 1
                            elif actual == 1 and predicted == 0:
                                stats["false_negatives"] += 1

                    if verbose:
                        print(f"Batch {batch_start//batch_size + 1}: {len(predictions)} samples | "
                              f"Attacks: {result['attack_count']} | Benign: {result['benign_count']} | "
                              f"Latency: {latency:.1f}ms ({result['avg_time_per_flow_ms']:.1f}ms/flow)")
                    else:
                        print(f"\rProgress: {min(batch_start + batch_size, len(indices))}/{len(indices)} samples", end="")
                else:
                    stats["errors"] += len(batch_indices)
                    print(f"\nBatch error: HTTP {response.status_code}")

            except Exception as e:
                stats["errors"] += len(batch_indices)
                print(f"\nBatch error: {e}")

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")


def print_statistics():
    """Print final simulation statistics."""
    global stats

    print(f"\n\n{'='*60}")
    print("SIMULATION RESULTS (Real CSE-CIC-IDS-2018 Test Data)")
    print(f"{'='*60}")

    total = stats["total_requests"]
    if total == 0:
        print("No requests completed.")
        return

    avg_latency = stats["total_latency_ms"] / total if total > 0 else 0

    print(f"\nTotal Samples Tested:  {total}")
    print(f"Errors:                {stats['errors']}")
    print(f"Average Latency:       {avg_latency:.2f}ms")

    print(f"\nPrediction Distribution:")
    print(f"  Benign Predicted:    {stats['benign_predicted']} ({100*stats['benign_predicted']/total:.1f}%)")
    print(f"  Attack Predicted:    {stats['attack_predicted']} ({100*stats['attack_predicted']/total:.1f}%)")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):   {stats['true_positives']:5d}  (Attack correctly detected)")
    print(f"  True Negatives (TN):   {stats['true_negatives']:5d}  (Benign correctly identified)")
    print(f"  False Positives (FP):  {stats['false_positives']:5d}  (Benign flagged as attack)")
    print(f"  False Negatives (FN):  {stats['false_negatives']:5d}  (Attack missed)")

    # Calculate metrics
    tp = stats["true_positives"]
    tn = stats["true_negatives"]
    fp = stats["false_positives"]
    fn = stats["false_negatives"]

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:      {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision:     {precision:.4f}  (Of predicted attacks, % actually attacks)")
    print(f"  Recall:        {recall:.4f}  (Of actual attacks, % detected)")
    print(f"  F1 Score:      {f1:.4f}")
    print(f"  Specificity:   {specificity:.4f}  (Of actual benign, % correctly identified)")
    print(f"  FPR:           {fpr:.4f}  (False Positive Rate)")

    print(f"\n{'='*60}")

    # Compare with expected
    print("\nComparison with Training Results:")
    print(f"  Expected Accuracy:  ~87.7%")
    print(f"  Actual Accuracy:    {accuracy*100:.2f}%")
    print(f"  Expected Recall:    ~79.6%")
    print(f"  Actual Recall:      {recall*100:.2f}%")

    if accuracy >= 0.85:
        print("\n✅ Model is performing as expected!")
    elif accuracy >= 0.75:
        print("\n⚠️ Model performance is slightly below expected.")
    else:
        print("\n❌ Model performance is significantly below expected. Check deployment.")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="IDS Real Data Simulator")
    parser.add_argument("--data-dir", type=str, default="processed_data",
                        help="Directory containing parquet files (default: processed_data)")
    parser.add_argument("--api", type=str, default="http://3.254.149.91:5000",
                        help="API URL")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to test (default: 100)")
    parser.add_argument("--mode", choices=["benign", "attack", "mixed"],
                        default="mixed", help="Test mode (default: mixed)")
    parser.add_argument("--rate", type=float, default=10,
                        help="Requests per second for single mode (default: 10)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Use batch endpoint with this batch size (default: 0 = single requests)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show each prediction result")

    args = parser.parse_args()

    run_simulation(
        args.data_dir,
        args.api,
        args.samples,
        args.mode,
        args.rate,
        args.verbose,
        args.batch_size
    )


if __name__ == "__main__":
    main()
