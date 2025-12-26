#!/usr/bin/env python3
"""
Enhanced Hint Server with A/B Testing Support

This server extends the original hint_server.py with:
- /mode/on  - Enable pacing (network-aware scheduling)
- /mode/off - Disable pacing (always return health=1.0)
- /mode/status - Get current mode

This allows A/B testing without restarting vLLM.

Usage:
    python hint_server_ab.py --port 5000
"""

import math
import os
import sys
import argparse
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# Import SmartTokenPacer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.predictive_health_monitor import SmartTokenPacer

app = Flask(__name__)

# ============================================================
# Global State
# ============================================================

class ServerState:
    def __init__(self):
        self.pacer = None
        self.data_path = "train_data.csv"
        
        # A/B Testing mode
        self.pacing_enabled = True  # True = Pacing ON, False = Pacing OFF
        self.mode_switch_count = 0
        self.mode_history = []
        
        # Session tracking for A/B experiments
        self.current_session_group = "pacing_on"
        self.session_requests = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()

STATE = ServerState()
ARGS = None

# ============================================================
# A/B Mode Control Endpoints
# ============================================================

@app.route("/mode/on", methods=["POST", "GET"])
def mode_on():
    """Enable network-aware pacing."""
    with STATE.lock:
        STATE.pacing_enabled = True
        STATE.current_session_group = "pacing_on"
        STATE.mode_switch_count += 1
        STATE.mode_history.append({
            "timestamp": datetime.now().isoformat(),
            "mode": "pacing_on"
        })
    return jsonify({
        "status": "ok",
        "mode": "pacing_on",
        "message": "Network-aware pacing ENABLED"
    })


@app.route("/mode/off", methods=["POST", "GET"])
def mode_off():
    """Disable pacing (baseline mode - full speed)."""
    with STATE.lock:
        STATE.pacing_enabled = False
        STATE.current_session_group = "pacing_off"
        STATE.mode_switch_count += 1
        STATE.mode_history.append({
            "timestamp": datetime.now().isoformat(),
            "mode": "pacing_off"
        })
    return jsonify({
        "status": "ok",
        "mode": "pacing_off",
        "message": "Pacing DISABLED (baseline mode, health=1.0)"
    })


@app.route("/mode/status", methods=["GET"])
def mode_status():
    """Get current mode status."""
    with STATE.lock:
        return jsonify({
            "pacing_enabled": STATE.pacing_enabled,
            "current_group": STATE.current_session_group,
            "mode_switches": STATE.mode_switch_count,
            "history": STATE.mode_history[-10:]  # Last 10 switches
        })


# ============================================================
# Main Hint Endpoint
# ============================================================

@app.route("/hint", methods=["GET"])
def get_hint():
    """
    Main hint endpoint - returns health score and token rate.
    
    In Pacing OFF mode, always returns health=1.0 (full speed).
    In Pacing ON mode, returns LSTM-predicted health score.
    """
    global STATE, ARGS
    
    try:
        # Increment request counter
        with STATE.lock:
            STATE.session_requests += 1
            pacing_enabled = STATE.pacing_enabled
            current_group = STATE.current_session_group
        
        # =====================================================
        # PACING OFF MODE - Always return full health
        # =====================================================
        if not pacing_enabled:
            return jsonify({
                "health": 1.0,
                "token_rate": 100.0,
                "pred_rtt": 0.0,
                "mode": "pacing_off",
                "metrics": {
                    "rtt": 0,
                    "cwnd": 0,
                    "throughput": 0,
                    "retrans": 0
                },
                "status": "baseline_mode"
            })
        
        # =====================================================
        # PACING ON MODE - Use LSTM prediction
        # =====================================================
        if STATE.pacer is None:
            return jsonify({"error": "Server initializing"}), 503

        if not os.path.exists(ARGS.data_path):
            return jsonify({
                "health": 1.0, 
                "token_rate": 50.0, 
                "mode": "pacing_on",
                "status": "waiting_for_data"
            })

        # Read latest data (v2 collector format)
        df = pd.read_csv(ARGS.data_path).tail(20)
        if df.empty:
            return jsonify({
                "health": 1.0, 
                "token_rate": 50.0, 
                "mode": "pacing_on",
                "status": "no_data"
            })
            
        latest = df.iloc[-1]
        
        # Prepare features for LSTM (7 features to match pretrained model):
        # [log_rtt, p95_rtt, avg_cwnd, throughput, retrans_count, rolling_avg_rtt, rtt_diff]
        current_rtt = latest['avg_rtt_us']
        log_rtt = np.log1p(max(current_rtt, 1.0))  # Prevent log(0)
        
        p95_rtt = latest.get('p95_rtt_us', current_rtt)
        avg_cwnd = latest.get('avg_cwnd', 0)
        throughput = latest.get('throughput_bps', 0)
        retrans_count = latest.get('retrans_count', 0)
        rolling_avg_rtt = latest.get('rolling_avg_rtt', current_rtt)
        
        # Calculate RTT diff
        rtt_diff = 0.0
        if len(df) >= 2:
            prev_rtt = df.iloc[-2]['avg_rtt_us']
            rtt_diff = log_rtt - np.log1p(max(prev_rtt, 1.0))
        
        # Build feature vector (7 features)
        features = [log_rtt, p95_rtt, avg_cwnd, throughput, retrans_count, rolling_avg_rtt, rtt_diff]
        
        # LSTM Step
        health, pred_rtt = STATE.pacer.step(features)
        
        # Safety floor - prevent deadlock
        health = max(0.1, float(health))
        
        # Calculate token rate from health
        base_rate = 5.0
        max_rate = 100.0
        factor = 1.0 / (1.0 + math.exp(-5.0 * (health - 0.5)))
        rate = round(base_rate + factor * (max_rate - base_rate), 2)
        
        return jsonify({
            "health": float(health),
            "token_rate": rate,
            "pred_rtt": float(pred_rtt),
            "mode": "pacing_on",
            "metrics": {
                "rtt": int(current_rtt),
                "cwnd": int(latest.get('avg_cwnd', 0)),
                "throughput": int(latest.get('throughput_bps', 0)),
                "retrans": int(latest.get('retrans_count', 0))
            }
        })
        
    except Exception as e:
        return jsonify({
            "health": 0.5, 
            "token_rate": 10.0, 
            "mode": "error",
            "error": str(e)
        })


# ============================================================
# Health Check
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Server health check endpoint."""
    return jsonify({
        "status": "healthy",
        "pacing_enabled": STATE.pacing_enabled,
        "uptime_requests": STATE.session_requests
    })


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM-based AI Hint Server with A/B Testing")
    parser.add_argument("--model-path", type=str, 
                       default="../model/final_online_model.pth", 
                       help="Path to LSTM model")
    parser.add_argument("--data-path", type=str, 
                       default="train_data.csv", 
                       help="Path to network data CSV (v2 format)")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Server port")
    parser.add_argument("--pacing-off", action="store_true",
                       help="Start in Pacing OFF mode (baseline)")
    parser.add_argument("--no-pretrain", action="store_true",
                       help="Don't load pretrained weights, use online learning only")
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    
    # Initialize pacing mode
    if ARGS.pacing_off:
        STATE.pacing_enabled = False
        STATE.current_session_group = "pacing_off"
    
    print("=" * 60)
    print("  eBPF-TokenFlow: A/B Hint Server")
    print("=" * 60)
    print(f"🔧 Configuration:")
    print(f"   LSTM Model:  {ARGS.model_path if not ARGS.no_pretrain else '(online learning)'}")
    print(f"   Data Path:   {ARGS.data_path}")
    print(f"   Port:        {ARGS.port}")
    print(f"   Mode:        {'Pacing OFF (baseline)' if not STATE.pacing_enabled else 'Pacing ON'}")
    print(f"   Pretrained:  {'No (online learning)' if ARGS.no_pretrain else 'Yes'}")
    print()
    print("📡 API Endpoints:")
    print(f"   GET  /hint        - Get health score & token rate")
    print(f"   POST /mode/on     - Enable network-aware pacing")
    print(f"   POST /mode/off    - Disable pacing (baseline)")
    print(f"   GET  /mode/status - Get current mode")
    print()

    # Initialize LSTM Pacer
    # Features (7 total, must match pretrained model):
    # [log_rtt, p95_rtt, avg_cwnd, throughput, retrans_count, rolling_avg_rtt, rtt_diff]
    NUM_FEATURES = 7
    
    model_path = None if ARGS.no_pretrain else ARGS.model_path
    
    if ARGS.no_pretrain:
        print("ℹ️  --no-pretrain flag set, using online learning mode")
    
    # Default scaler parameters for 7 features
    # These are approximate values based on typical network metrics
    default_mean = [10.0, 50000.0, 100.0, 1000000.0, 0.0, 50000.0, 0.0]
    default_scale = [2.0, 50000.0, 100.0, 1000000.0, 1.0, 50000.0, 1.0]
    
    try:
        # First try to load with pretrained weights (if not disabled)
        STATE.pacer = SmartTokenPacer(model_path=model_path, input_features=NUM_FEATURES)
        STATE.pacer.set_scaler(mean=default_mean, scale=default_scale)
        if model_path:
            print("✅ LSTM model loaded successfully (7 features)")
        else:
            print("✅ LSTM initialized in online learning mode (7 features)")
    except Exception as e:
        print(f"⚠️  Failed to load pretrained model: {e}")
        print("   ➡️  Falling back to online learning mode (no pretrained weights)")
        try:
            # Initialize without pretrained weights - will learn online
            STATE.pacer = SmartTokenPacer(model_path=None, input_features=NUM_FEATURES)
            STATE.pacer.set_scaler(mean=default_mean, scale=default_scale)
            print("✅ LSTM initialized in online learning mode (7 features)")
        except Exception as e2:
            print(f"❌ Failed to initialize LSTM: {e2}")
            print("   Server will still run, but Pacing ON mode may not work")

    print()
    print(f"🚀 Starting server on port {ARGS.port}...")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=ARGS.port, threaded=True)

