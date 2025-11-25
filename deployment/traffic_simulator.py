#!/usr/bin/env python3
"""
IDS Traffic Simulator - End-to-End Testing Framework
Simulates various network traffic patterns (benign and attack) for testing the IDS API.

Usage:
    python traffic_simulator.py --mode all --duration 60 --api http://3.254.149.91:5000
"""

import requests
import json
import time
import random
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

# API endpoint
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


def get_base_features():
    """Return base feature dictionary with all 69 features set to 0."""
    return {
        "Protocol": 0,
        "Flow Duration": 0,
        "Total Fwd Packets": 0,
        "Total Backward Packets": 0,
        "Fwd Packets Length Total": 0,
        "Bwd Packets Length Total": 0,
        "Fwd Packet Length Max": 0,
        "Fwd Packet Length Min": 0,
        "Fwd Packet Length Mean": 0,
        "Fwd Packet Length Std": 0,
        "Bwd Packet Length Max": 0,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 0,
        "Bwd Packet Length Std": 0,
        "Flow Bytes/s": 0,
        "Flow Packets/s": 0,
        "Flow IAT Mean": 0,
        "Flow IAT Std": 0,
        "Flow IAT Max": 0,
        "Flow IAT Min": 0,
        "Fwd IAT Total": 0,
        "Fwd IAT Mean": 0,
        "Fwd IAT Std": 0,
        "Fwd IAT Max": 0,
        "Fwd IAT Min": 0,
        "Bwd IAT Total": 0,
        "Bwd IAT Mean": 0,
        "Bwd IAT Std": 0,
        "Bwd IAT Max": 0,
        "Bwd IAT Min": 0,
        "Fwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Fwd Header Length": 0,
        "Bwd Header Length": 0,
        "Fwd Packets/s": 0,
        "Bwd Packets/s": 0,
        "Packet Length Min": 0,
        "Packet Length Max": 0,
        "Packet Length Mean": 0,
        "Packet Length Std": 0,
        "Packet Length Variance": 0,
        "FIN Flag Count": 0,
        "SYN Flag Count": 0,
        "RST Flag Count": 0,
        "PSH Flag Count": 0,
        "ACK Flag Count": 0,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0,
        "Down/Up Ratio": 0,
        "Avg Packet Size": 0,
        "Avg Fwd Segment Size": 0,
        "Avg Bwd Segment Size": 0,
        "Subflow Fwd Packets": 0,
        "Subflow Fwd Bytes": 0,
        "Subflow Bwd Packets": 0,
        "Subflow Bwd Bytes": 0,
        "Init Fwd Win Bytes": 0,
        "Init Bwd Win Bytes": 0,
        "Fwd Act Data Packets": 0,
        "Fwd Seg Size Min": 0,
        "Active Mean": 0,
        "Active Std": 0,
        "Active Max": 0,
        "Active Min": 0,
        "Idle Mean": 0,
        "Idle Std": 0,
        "Idle Max": 0,
        "Idle Min": 0,
    }


# ============================================================================
# BENIGN TRAFFIC GENERATORS
# ============================================================================

def generate_http_browsing():
    """Normal HTTP web browsing traffic."""
    features = get_base_features()
    features.update({
        "Protocol": 6,  # TCP
        "Flow Duration": random.randint(100000, 5000000),
        "Total Fwd Packets": random.randint(5, 30),
        "Total Backward Packets": random.randint(5, 50),
        "Fwd Packets Length Total": random.randint(500, 5000),
        "Bwd Packets Length Total": random.randint(5000, 500000),
        "Fwd Packet Length Max": random.randint(100, 1460),
        "Fwd Packet Length Min": 0,
        "Fwd Packet Length Mean": random.uniform(50, 500),
        "Fwd Packet Length Std": random.uniform(50, 300),
        "Bwd Packet Length Max": 1460,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": random.uniform(200, 1000),
        "Bwd Packet Length Std": random.uniform(100, 500),
        "Flow Bytes/s": random.uniform(10000, 500000),
        "Flow Packets/s": random.uniform(10, 200),
        "Flow IAT Mean": random.uniform(5000, 50000),
        "Flow IAT Std": random.uniform(10000, 100000),
        "Flow IAT Max": random.randint(50000, 500000),
        "Flow IAT Min": random.randint(0, 1000),
        "Fwd IAT Total": random.randint(50000, 2000000),
        "Fwd IAT Mean": random.uniform(5000, 100000),
        "Fwd IAT Std": random.uniform(10000, 200000),
        "Fwd IAT Max": random.randint(50000, 500000),
        "Fwd IAT Min": random.randint(0, 5000),
        "Bwd IAT Total": random.randint(50000, 2000000),
        "Bwd IAT Mean": random.uniform(5000, 80000),
        "Bwd IAT Std": random.uniform(10000, 150000),
        "Bwd IAT Max": random.randint(50000, 400000),
        "Bwd IAT Min": random.randint(0, 5000),
        "Fwd PSH Flags": random.randint(0, 5),
        "Fwd Header Length": random.randint(200, 800),
        "Bwd Header Length": random.randint(200, 1000),
        "Fwd Packets/s": random.uniform(5, 100),
        "Bwd Packets/s": random.uniform(5, 150),
        "Packet Length Min": 0,
        "Packet Length Max": 1460,
        "Packet Length Mean": random.uniform(100, 800),
        "Packet Length Std": random.uniform(100, 500),
        "Packet Length Variance": random.uniform(10000, 250000),
        "FIN Flag Count": 1,
        "SYN Flag Count": 1,
        "ACK Flag Count": random.randint(5, 50),
        "PSH Flag Count": random.randint(1, 10),
        "Down/Up Ratio": random.uniform(0.5, 10),
        "Avg Packet Size": random.uniform(100, 800),
        "Avg Fwd Segment Size": random.uniform(50, 500),
        "Avg Bwd Segment Size": random.uniform(200, 1000),
        "Subflow Fwd Packets": random.randint(5, 30),
        "Subflow Fwd Bytes": random.randint(500, 5000),
        "Subflow Bwd Packets": random.randint(5, 50),
        "Subflow Bwd Bytes": random.randint(5000, 500000),
        "Init Fwd Win Bytes": random.randint(16384, 65535),
        "Init Bwd Win Bytes": random.randint(16384, 65535),
        "Fwd Act Data Packets": random.randint(1, 20),
        "Active Mean": random.uniform(1000, 50000),
        "Active Std": random.uniform(500, 20000),
        "Active Max": random.randint(5000, 100000),
        "Active Min": random.randint(100, 5000),
        "Idle Mean": random.uniform(10000, 200000),
        "Idle Std": random.uniform(5000, 100000),
        "Idle Max": random.randint(50000, 500000),
        "Idle Min": random.randint(1000, 50000),
    })
    return features, "benign", "HTTP Browsing"


def generate_https_traffic():
    """Normal HTTPS encrypted traffic."""
    features = get_base_features()
    features.update({
        "Protocol": 6,
        "Flow Duration": random.randint(500000, 10000000),
        "Total Fwd Packets": random.randint(10, 100),
        "Total Backward Packets": random.randint(10, 150),
        "Fwd Packets Length Total": random.randint(1000, 20000),
        "Bwd Packets Length Total": random.randint(10000, 1000000),
        "Fwd Packet Length Max": 1460,
        "Fwd Packet Length Min": 0,
        "Fwd Packet Length Mean": random.uniform(100, 800),
        "Fwd Packet Length Std": random.uniform(100, 500),
        "Bwd Packet Length Max": 1460,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": random.uniform(500, 1200),
        "Bwd Packet Length Std": random.uniform(200, 600),
        "Flow Bytes/s": random.uniform(50000, 2000000),
        "Flow Packets/s": random.uniform(20, 500),
        "Flow IAT Mean": random.uniform(2000, 30000),
        "Flow IAT Std": random.uniform(5000, 50000),
        "Flow IAT Max": random.randint(20000, 200000),
        "Flow IAT Min": random.randint(0, 500),
        "Fwd IAT Total": random.randint(100000, 5000000),
        "Fwd IAT Mean": random.uniform(2000, 50000),
        "Fwd IAT Std": random.uniform(5000, 100000),
        "Fwd IAT Max": random.randint(20000, 300000),
        "Fwd IAT Min": random.randint(0, 2000),
        "Bwd IAT Total": random.randint(100000, 5000000),
        "Bwd IAT Mean": random.uniform(2000, 40000),
        "Bwd IAT Std": random.uniform(5000, 80000),
        "Bwd IAT Max": random.randint(20000, 250000),
        "Bwd IAT Min": random.randint(0, 2000),
        "Fwd Header Length": random.randint(400, 1200),
        "Bwd Header Length": random.randint(400, 1500),
        "Fwd Packets/s": random.uniform(10, 200),
        "Bwd Packets/s": random.uniform(10, 300),
        "Packet Length Min": 0,
        "Packet Length Max": 1460,
        "Packet Length Mean": random.uniform(300, 1000),
        "Packet Length Std": random.uniform(200, 600),
        "Packet Length Variance": random.uniform(40000, 360000),
        "FIN Flag Count": 1,
        "SYN Flag Count": 1,
        "ACK Flag Count": random.randint(10, 100),
        "PSH Flag Count": random.randint(5, 30),
        "Down/Up Ratio": random.uniform(1, 20),
        "Avg Packet Size": random.uniform(300, 1000),
        "Avg Fwd Segment Size": random.uniform(100, 800),
        "Avg Bwd Segment Size": random.uniform(500, 1200),
        "Subflow Fwd Packets": random.randint(10, 100),
        "Subflow Fwd Bytes": random.randint(1000, 20000),
        "Subflow Bwd Packets": random.randint(10, 150),
        "Subflow Bwd Bytes": random.randint(10000, 1000000),
        "Init Fwd Win Bytes": 65535,
        "Init Bwd Win Bytes": 65535,
        "Fwd Act Data Packets": random.randint(5, 50),
        "Active Mean": random.uniform(2000, 100000),
        "Active Std": random.uniform(1000, 50000),
        "Active Max": random.randint(10000, 200000),
        "Active Min": random.randint(500, 10000),
        "Idle Mean": random.uniform(20000, 500000),
        "Idle Std": random.uniform(10000, 200000),
        "Idle Max": random.randint(100000, 1000000),
        "Idle Min": random.randint(5000, 100000),
    })
    return features, "benign", "HTTPS Traffic"


def generate_dns_query():
    """Normal DNS query traffic."""
    features = get_base_features()
    features.update({
        "Protocol": 17,  # UDP
        "Flow Duration": random.randint(1000, 50000),
        "Total Fwd Packets": 1,
        "Total Backward Packets": 1,
        "Fwd Packets Length Total": random.randint(40, 100),
        "Bwd Packets Length Total": random.randint(50, 500),
        "Fwd Packet Length Max": random.randint(40, 100),
        "Fwd Packet Length Min": random.randint(40, 100),
        "Fwd Packet Length Mean": random.uniform(40, 100),
        "Fwd Packet Length Std": 0,
        "Bwd Packet Length Max": random.randint(50, 500),
        "Bwd Packet Length Min": random.randint(50, 500),
        "Bwd Packet Length Mean": random.uniform(50, 500),
        "Bwd Packet Length Std": 0,
        "Flow Bytes/s": random.uniform(1000, 50000),
        "Flow Packets/s": random.uniform(20, 200),
        "Flow IAT Mean": random.uniform(1000, 30000),
        "Flow IAT Std": 0,
        "Flow IAT Max": random.randint(1000, 50000),
        "Flow IAT Min": random.randint(1000, 50000),
        "Packet Length Min": random.randint(40, 50),
        "Packet Length Max": random.randint(50, 500),
        "Packet Length Mean": random.uniform(50, 300),
        "Down/Up Ratio": random.uniform(1, 10),
        "Avg Packet Size": random.uniform(50, 300),
        "Init Fwd Win Bytes": 0,
        "Init Bwd Win Bytes": 0,
    })
    return features, "benign", "DNS Query"


# ============================================================================
# ATTACK TRAFFIC GENERATORS
# ============================================================================

def generate_syn_flood():
    """SYN Flood DDoS attack pattern."""
    features = get_base_features()
    features.update({
        "Protocol": 6,
        "Flow Duration": random.randint(100, 5000),
        "Total Fwd Packets": random.randint(50, 500),
        "Total Backward Packets": 0,
        "Fwd Packets Length Total": random.randint(2000, 20000),
        "Bwd Packets Length Total": 0,
        "Fwd Packet Length Max": 60,
        "Fwd Packet Length Min": 40,
        "Fwd Packet Length Mean": random.uniform(40, 60),
        "Fwd Packet Length Std": random.uniform(0, 10),
        "Bwd Packet Length Max": 0,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 0,
        "Bwd Packet Length Std": 0,
        "Flow Bytes/s": random.uniform(1000000, 50000000),
        "Flow Packets/s": random.uniform(10000, 500000),
        "Flow IAT Mean": random.uniform(1, 100),
        "Flow IAT Std": random.uniform(0, 50),
        "Flow IAT Max": random.randint(10, 500),
        "Flow IAT Min": 0,
        "Fwd IAT Total": random.randint(100, 5000),
        "Fwd IAT Mean": random.uniform(1, 50),
        "Fwd IAT Std": random.uniform(0, 30),
        "Fwd IAT Max": random.randint(10, 200),
        "Fwd IAT Min": 0,
        "Bwd IAT Total": 0,
        "Bwd IAT Mean": 0,
        "Bwd IAT Std": 0,
        "Bwd IAT Max": 0,
        "Bwd IAT Min": 0,
        "Fwd Header Length": random.randint(1000, 10000),
        "Bwd Header Length": 0,
        "Fwd Packets/s": random.uniform(10000, 500000),
        "Bwd Packets/s": 0,
        "Packet Length Min": 40,
        "Packet Length Max": 60,
        "Packet Length Mean": random.uniform(40, 60),
        "Packet Length Std": random.uniform(0, 10),
        "Packet Length Variance": random.uniform(0, 100),
        "SYN Flag Count": random.randint(50, 500),
        "ACK Flag Count": 0,
        "FIN Flag Count": 0,
        "Down/Up Ratio": 0,
        "Avg Packet Size": random.uniform(40, 60),
        "Avg Fwd Segment Size": random.uniform(40, 60),
        "Avg Bwd Segment Size": 0,
        "Subflow Fwd Packets": random.randint(50, 500),
        "Subflow Fwd Bytes": random.randint(2000, 20000),
        "Subflow Bwd Packets": 0,
        "Subflow Bwd Bytes": 0,
        "Init Fwd Win Bytes": random.randint(1024, 4096),
        "Init Bwd Win Bytes": 0,
        "Fwd Act Data Packets": 0,
        "Fwd Seg Size Min": 40,
    })
    return features, "attack", "SYN Flood"


def generate_port_scan():
    """Port scanning attack pattern."""
    features = get_base_features()
    features.update({
        "Protocol": 6,
        "Flow Duration": random.randint(50, 1000),
        "Total Fwd Packets": random.randint(1, 5),
        "Total Backward Packets": random.randint(0, 2),
        "Fwd Packets Length Total": random.randint(40, 200),
        "Bwd Packets Length Total": random.randint(0, 100),
        "Fwd Packet Length Max": random.randint(40, 60),
        "Fwd Packet Length Min": random.randint(40, 60),
        "Fwd Packet Length Mean": random.uniform(40, 60),
        "Fwd Packet Length Std": 0,
        "Bwd Packet Length Max": random.randint(0, 60),
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": random.uniform(0, 60),
        "Bwd Packet Length Std": 0,
        "Flow Bytes/s": random.uniform(10000, 500000),
        "Flow Packets/s": random.uniform(1000, 50000),
        "Flow IAT Mean": random.uniform(10, 500),
        "Flow IAT Std": random.uniform(0, 200),
        "Flow IAT Max": random.randint(50, 1000),
        "Flow IAT Min": 0,
        "Fwd IAT Total": random.randint(50, 1000),
        "Fwd IAT Mean": random.uniform(10, 200),
        "Fwd IAT Std": random.uniform(0, 100),
        "Fwd IAT Max": random.randint(50, 500),
        "Fwd IAT Min": 0,
        "Fwd Header Length": random.randint(40, 200),
        "Bwd Header Length": random.randint(0, 100),
        "Fwd Packets/s": random.uniform(1000, 50000),
        "Bwd Packets/s": random.uniform(0, 10000),
        "Packet Length Min": 40,
        "Packet Length Max": 60,
        "Packet Length Mean": random.uniform(40, 60),
        "Packet Length Std": random.uniform(0, 10),
        "SYN Flag Count": random.randint(1, 5),
        "RST Flag Count": random.randint(0, 3),
        "ACK Flag Count": random.randint(0, 2),
        "Down/Up Ratio": random.uniform(0, 1),
        "Avg Packet Size": random.uniform(40, 60),
        "Init Fwd Win Bytes": random.randint(1024, 8192),
        "Init Bwd Win Bytes": 0,
    })
    return features, "attack", "Port Scan"


def generate_ssh_brute_force():
    """SSH brute force attack pattern."""
    features = get_base_features()
    features.update({
        "Protocol": 6,
        "Flow Duration": random.randint(10000, 100000),
        "Total Fwd Packets": random.randint(10, 50),
        "Total Backward Packets": random.randint(10, 50),
        "Fwd Packets Length Total": random.randint(500, 5000),
        "Bwd Packets Length Total": random.randint(500, 5000),
        "Fwd Packet Length Max": random.randint(100, 500),
        "Fwd Packet Length Min": random.randint(20, 50),
        "Fwd Packet Length Mean": random.uniform(50, 200),
        "Fwd Packet Length Std": random.uniform(20, 100),
        "Bwd Packet Length Max": random.randint(100, 500),
        "Bwd Packet Length Min": random.randint(20, 50),
        "Bwd Packet Length Mean": random.uniform(50, 200),
        "Bwd Packet Length Std": random.uniform(20, 100),
        "Flow Bytes/s": random.uniform(5000, 100000),
        "Flow Packets/s": random.uniform(100, 2000),
        "Flow IAT Mean": random.uniform(500, 5000),
        "Flow IAT Std": random.uniform(200, 2000),
        "Flow IAT Max": random.randint(2000, 20000),
        "Flow IAT Min": random.randint(0, 500),
        "Fwd IAT Total": random.randint(5000, 50000),
        "Fwd IAT Mean": random.uniform(200, 2000),
        "Fwd IAT Std": random.uniform(100, 1000),
        "Fwd IAT Max": random.randint(1000, 10000),
        "Fwd IAT Min": random.randint(0, 200),
        "Bwd IAT Total": random.randint(5000, 50000),
        "Bwd IAT Mean": random.uniform(200, 2000),
        "Bwd IAT Std": random.uniform(100, 1000),
        "Bwd IAT Max": random.randint(1000, 10000),
        "Bwd IAT Min": random.randint(0, 200),
        "Fwd PSH Flags": random.randint(5, 30),
        "Fwd Header Length": random.randint(200, 1000),
        "Bwd Header Length": random.randint(200, 1000),
        "Fwd Packets/s": random.uniform(100, 2000),
        "Bwd Packets/s": random.uniform(100, 2000),
        "Packet Length Min": 20,
        "Packet Length Max": random.randint(200, 500),
        "Packet Length Mean": random.uniform(50, 200),
        "Packet Length Std": random.uniform(30, 150),
        "SYN Flag Count": 1,
        "ACK Flag Count": random.randint(10, 50),
        "PSH Flag Count": random.randint(5, 30),
        "Down/Up Ratio": random.uniform(0.8, 1.2),
        "Avg Packet Size": random.uniform(50, 200),
        "Init Fwd Win Bytes": random.randint(16384, 65535),
        "Init Bwd Win Bytes": random.randint(16384, 65535),
        "Fwd Act Data Packets": random.randint(5, 30),
    })
    return features, "attack", "SSH Brute Force"


def generate_http_flood():
    """HTTP flood DDoS attack pattern."""
    features = get_base_features()
    features.update({
        "Protocol": 6,
        "Flow Duration": random.randint(1000, 10000),
        "Total Fwd Packets": random.randint(20, 100),
        "Total Backward Packets": random.randint(0, 5),
        "Fwd Packets Length Total": random.randint(5000, 50000),
        "Bwd Packets Length Total": random.randint(0, 1000),
        "Fwd Packet Length Max": random.randint(500, 1460),
        "Fwd Packet Length Min": random.randint(100, 300),
        "Fwd Packet Length Mean": random.uniform(200, 800),
        "Fwd Packet Length Std": random.uniform(50, 300),
        "Bwd Packet Length Max": random.randint(0, 200),
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": random.uniform(0, 100),
        "Bwd Packet Length Std": random.uniform(0, 50),
        "Flow Bytes/s": random.uniform(500000, 10000000),
        "Flow Packets/s": random.uniform(5000, 100000),
        "Flow IAT Mean": random.uniform(10, 200),
        "Flow IAT Std": random.uniform(5, 100),
        "Flow IAT Max": random.randint(50, 500),
        "Flow IAT Min": 0,
        "Fwd IAT Total": random.randint(500, 5000),
        "Fwd IAT Mean": random.uniform(5, 100),
        "Fwd IAT Std": random.uniform(2, 50),
        "Fwd IAT Max": random.randint(20, 200),
        "Fwd IAT Min": 0,
        "Fwd PSH Flags": random.randint(10, 50),
        "Fwd Header Length": random.randint(500, 3000),
        "Bwd Header Length": random.randint(0, 200),
        "Fwd Packets/s": random.uniform(5000, 100000),
        "Bwd Packets/s": random.uniform(0, 1000),
        "Packet Length Min": 100,
        "Packet Length Max": 1460,
        "Packet Length Mean": random.uniform(200, 800),
        "Packet Length Std": random.uniform(100, 400),
        "SYN Flag Count": 1,
        "ACK Flag Count": random.randint(20, 100),
        "PSH Flag Count": random.randint(10, 50),
        "Down/Up Ratio": random.uniform(0, 0.2),
        "Avg Packet Size": random.uniform(200, 800),
        "Avg Fwd Segment Size": random.uniform(200, 800),
        "Init Fwd Win Bytes": random.randint(16384, 65535),
        "Init Bwd Win Bytes": 0,
        "Fwd Act Data Packets": random.randint(10, 50),
    })
    return features, "attack", "HTTP Flood"


def generate_slowloris():
    """Slowloris attack pattern - slow HTTP attack."""
    features = get_base_features()
    features.update({
        "Protocol": 6,
        "Flow Duration": random.randint(10000000, 100000000),  # Very long duration
        "Total Fwd Packets": random.randint(100, 500),
        "Total Backward Packets": random.randint(0, 10),
        "Fwd Packets Length Total": random.randint(5000, 20000),
        "Bwd Packets Length Total": random.randint(0, 500),
        "Fwd Packet Length Max": random.randint(50, 200),
        "Fwd Packet Length Min": random.randint(10, 50),
        "Fwd Packet Length Mean": random.uniform(30, 100),
        "Fwd Packet Length Std": random.uniform(10, 50),
        "Bwd Packet Length Max": random.randint(0, 100),
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": random.uniform(0, 50),
        "Bwd Packet Length Std": 0,
        "Flow Bytes/s": random.uniform(10, 500),  # Very low bytes/s
        "Flow Packets/s": random.uniform(0.1, 10),  # Very low packets/s
        "Flow IAT Mean": random.uniform(100000, 1000000),  # Long intervals
        "Flow IAT Std": random.uniform(50000, 500000),
        "Flow IAT Max": random.randint(500000, 5000000),
        "Flow IAT Min": random.randint(10000, 100000),
        "Fwd IAT Total": random.randint(10000000, 100000000),
        "Fwd IAT Mean": random.uniform(100000, 1000000),
        "Fwd IAT Std": random.uniform(50000, 500000),
        "Fwd IAT Max": random.randint(500000, 5000000),
        "Fwd IAT Min": random.randint(10000, 100000),
        "Fwd Header Length": random.randint(2000, 10000),
        "Bwd Header Length": random.randint(0, 200),
        "Fwd Packets/s": random.uniform(0.1, 10),
        "Bwd Packets/s": random.uniform(0, 1),
        "Packet Length Min": 10,
        "Packet Length Max": 200,
        "Packet Length Mean": random.uniform(30, 100),
        "Packet Length Std": random.uniform(10, 50),
        "SYN Flag Count": 1,
        "ACK Flag Count": random.randint(50, 200),
        "PSH Flag Count": random.randint(50, 200),
        "Down/Up Ratio": 0,
        "Avg Packet Size": random.uniform(30, 100),
        "Init Fwd Win Bytes": random.randint(16384, 65535),
        "Init Bwd Win Bytes": 0,
        "Fwd Act Data Packets": random.randint(50, 200),
        "Idle Mean": random.uniform(500000, 5000000),  # Long idle times
        "Idle Std": random.uniform(200000, 2000000),
        "Idle Max": random.randint(1000000, 10000000),
        "Idle Min": random.randint(100000, 1000000),
    })
    return features, "attack", "Slowloris"


def generate_udp_flood():
    """UDP flood DDoS attack pattern."""
    features = get_base_features()
    features.update({
        "Protocol": 17,  # UDP
        "Flow Duration": random.randint(100, 5000),
        "Total Fwd Packets": random.randint(100, 1000),
        "Total Backward Packets": 0,
        "Fwd Packets Length Total": random.randint(10000, 100000),
        "Bwd Packets Length Total": 0,
        "Fwd Packet Length Max": random.randint(500, 1400),
        "Fwd Packet Length Min": random.randint(100, 500),
        "Fwd Packet Length Mean": random.uniform(200, 800),
        "Fwd Packet Length Std": random.uniform(50, 300),
        "Bwd Packet Length Max": 0,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 0,
        "Bwd Packet Length Std": 0,
        "Flow Bytes/s": random.uniform(5000000, 100000000),
        "Flow Packets/s": random.uniform(50000, 1000000),
        "Flow IAT Mean": random.uniform(0.5, 10),
        "Flow IAT Std": random.uniform(0, 5),
        "Flow IAT Max": random.randint(5, 50),
        "Flow IAT Min": 0,
        "Fwd IAT Total": random.randint(100, 5000),
        "Fwd IAT Mean": random.uniform(0.5, 10),
        "Fwd IAT Std": random.uniform(0, 5),
        "Fwd IAT Max": random.randint(5, 50),
        "Fwd IAT Min": 0,
        "Fwd Header Length": random.randint(800, 8000),
        "Bwd Header Length": 0,
        "Fwd Packets/s": random.uniform(50000, 1000000),
        "Bwd Packets/s": 0,
        "Packet Length Min": 100,
        "Packet Length Max": 1400,
        "Packet Length Mean": random.uniform(200, 800),
        "Packet Length Std": random.uniform(50, 300),
        "Down/Up Ratio": 0,
        "Avg Packet Size": random.uniform(200, 800),
        "Avg Fwd Segment Size": random.uniform(200, 800),
        "Avg Bwd Segment Size": 0,
        "Subflow Fwd Packets": random.randint(100, 1000),
        "Subflow Fwd Bytes": random.randint(10000, 100000),
        "Subflow Bwd Packets": 0,
        "Subflow Bwd Bytes": 0,
    })
    return features, "attack", "UDP Flood"


# Traffic generator mapping
TRAFFIC_GENERATORS = {
    "benign": [
        generate_http_browsing,
        generate_https_traffic,
        generate_dns_query,
    ],
    "attack": [
        generate_syn_flood,
        generate_port_scan,
        generate_ssh_brute_force,
        generate_http_flood,
        generate_slowloris,
        generate_udp_flood,
    ]
}


def send_prediction(features, actual_label, attack_type, api_url):
    """Send a prediction request to the API."""
    global stats

    try:
        start_time = time.time()
        response = requests.post(
            f"{api_url}/predict",
            json={"features": features},
            timeout=10
        )
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            predicted_label = "attack" if result["prediction"] == 1 else "benign"

            with stats_lock:
                stats["total_requests"] += 1
                stats["total_latency_ms"] += latency

                if predicted_label == "attack":
                    stats["attack_predicted"] += 1
                else:
                    stats["benign_predicted"] += 1

                # Calculate confusion matrix
                if actual_label == "attack" and predicted_label == "attack":
                    stats["true_positives"] += 1
                elif actual_label == "benign" and predicted_label == "attack":
                    stats["false_positives"] += 1
                elif actual_label == "benign" and predicted_label == "benign":
                    stats["true_negatives"] += 1
                elif actual_label == "attack" and predicted_label == "benign":
                    stats["false_negatives"] += 1

            return {
                "success": True,
                "actual": actual_label,
                "predicted": predicted_label,
                "attack_type": attack_type,
                "confidence": result["confidence"],
                "latency_ms": latency
            }
        else:
            with stats_lock:
                stats["errors"] += 1
            return {"success": False, "error": response.text}

    except Exception as e:
        with stats_lock:
            stats["errors"] += 1
        return {"success": False, "error": str(e)}


def run_simulation(mode, duration, rate, api_url, verbose):
    """Run the traffic simulation."""
    global stats

    print(f"\n{'='*60}")
    print(f"IDS Traffic Simulator")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Duration: {duration} seconds")
    print(f"Rate: {rate} requests/second")
    print(f"API: {api_url}")
    print(f"{'='*60}\n")

    # Select generators based on mode
    if mode == "benign":
        generators = TRAFFIC_GENERATORS["benign"]
        weights = [1] * len(generators)
    elif mode == "attack":
        generators = TRAFFIC_GENERATORS["attack"]
        weights = [1] * len(generators)
    else:  # mixed or all
        generators = TRAFFIC_GENERATORS["benign"] + TRAFFIC_GENERATORS["attack"]
        # 50% benign, 50% attack
        benign_weight = 1.0 / len(TRAFFIC_GENERATORS["benign"])
        attack_weight = 1.0 / len(TRAFFIC_GENERATORS["attack"])
        weights = [benign_weight] * len(TRAFFIC_GENERATORS["benign"]) + \
                  [attack_weight] * len(TRAFFIC_GENERATORS["attack"])

    start_time = time.time()
    request_count = 0
    interval = 1.0 / rate

    print("Starting simulation...")
    print(f"Press Ctrl+C to stop early\n")

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            while time.time() - start_time < duration:
                # Select random generator
                generator = random.choices(generators, weights=weights)[0]
                features, actual_label, attack_type = generator()

                # Submit prediction request
                future = executor.submit(
                    send_prediction, features, actual_label, attack_type, api_url
                )

                if verbose:
                    result = future.result()
                    if result["success"]:
                        status = "✓" if result["actual"] == result["predicted"] else "✗"
                        print(f"{status} {result['attack_type']:20s} | "
                              f"Actual: {result['actual']:6s} | "
                              f"Predicted: {result['predicted']:6s} | "
                              f"Confidence: {result['confidence']:.2f} | "
                              f"Latency: {result['latency_ms']:.1f}ms")

                request_count += 1

                # Progress update every 10 requests
                if not verbose and request_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"\rRequests: {request_count} | "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"Rate: {request_count/elapsed:.1f}/s", end="")

                time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")

    # Print final statistics
    print_statistics()


def print_statistics():
    """Print final simulation statistics."""
    global stats

    print(f"\n\n{'='*60}")
    print("SIMULATION RESULTS")
    print(f"{'='*60}")

    total = stats["total_requests"]
    if total == 0:
        print("No requests completed.")
        return

    avg_latency = stats["total_latency_ms"] / total if total > 0 else 0

    print(f"\nTotal Requests:     {total}")
    print(f"Errors:             {stats['errors']}")
    print(f"Average Latency:    {avg_latency:.2f}ms")

    print(f"\nPrediction Distribution:")
    print(f"  Benign:           {stats['benign_predicted']} ({100*stats['benign_predicted']/total:.1f}%)")
    print(f"  Attack:           {stats['attack_predicted']} ({100*stats['attack_predicted']/total:.1f}%)")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:   {stats['true_positives']}")
    print(f"  True Negatives:   {stats['true_negatives']}")
    print(f"  False Positives:  {stats['false_positives']}")
    print(f"  False Negatives:  {stats['false_negatives']}")

    # Calculate metrics
    tp = stats["true_positives"]
    tn = stats["true_negatives"]
    fp = stats["false_positives"]
    fn = stats["false_negatives"]

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1 Score:         {f1:.4f}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="IDS Traffic Simulator")
    parser.add_argument("--mode", choices=["benign", "attack", "mixed", "all"],
                        default="mixed", help="Traffic mode (default: mixed)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Simulation duration in seconds (default: 60)")
    parser.add_argument("--rate", type=float, default=5,
                        help="Requests per second (default: 5)")
    parser.add_argument("--api", type=str, default="http://3.254.149.91:5000",
                        help="API URL")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show each prediction result")

    args = parser.parse_args()

    run_simulation(args.mode, args.duration, args.rate, args.api, args.verbose)


if __name__ == "__main__":
    main()
