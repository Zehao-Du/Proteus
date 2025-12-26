#!/usr/bin/env python3
import math
import pandas as pd
import time
import sys
import os
import argparse
import torch
import numpy as np
from flask import Flask, jsonify

# Import the SmartTokenPacer from the model directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.predictive_health_monitor import SmartTokenPacer

app = Flask(__name__)

# Global variables placeholder
pacer = None
ARGS = None 

@app.route("/hint", methods=["GET"])
def get_hint():
    global pacer, ARGS
    try:
        if pacer is None:
             return jsonify({"error": "Server initializing"}), 503

        if not os.path.exists(ARGS.data_path):
             return jsonify({"health": 1.0, "token_rate": 50.0, "status": "waiting_for_data"})

        # Read latest data (v2 collector format)
        df = pd.read_csv(ARGS.data_path).tail(20)
        if df.empty:
            return jsonify({"health": 1.0, "token_rate": 50.0, "status": "no_data"})
            
        latest = df.iloc[-1]
        
        # Prepare features for LSTM: [log_rtt, rtt_diff] 
        current_rtt = latest['avg_rtt_us']
        log_rtt = np.log1p(current_rtt)
        
        rtt_diff = 0.0
        if len(df) >= 2:
            prev_rtt = df.iloc[-2]['avg_rtt_us']
            rtt_diff = log_rtt - np.log1p(prev_rtt)
        
        # LSTM Step
        health, pred_rtt = pacer.step([log_rtt, rtt_diff])
        
        # 修复点：健康度兜底，防止 Health=0 导致 vLLM 调度死锁
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
            "metrics": {
                "rtt": int(current_rtt),
                "cwnd": int(latest.get('avg_cwnd', 0)),
                "throughput": int(latest.get('throughput_bps', 0)),
                "retrans": int(latest.get('retrans_count', 0))
            }
        })
    except Exception as e:
        return jsonify({"health": 0.5, "token_rate": 10.0, "error": str(e)})

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM-based AI Hint Server")
    parser.add_argument("--model-path", type=str, default="../model/final_online_model.pth", help="Path to LSTM model")
    parser.add_argument("--data-path", type=str, default="train_data.csv", help="Path to network data CSV (v2 format)")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    return parser.parse_args()

if __name__ == "__main__":
    ARGS = parse_args()
    
    print(f"🔧 Configuration:")
    print(f"   LSTM Model: {ARGS.model_path}")
    print(f"   Data Path:  {ARGS.data_path}")

    # Initialize LSTM Pacer
    pacer = SmartTokenPacer(model_path=ARGS.model_path, input_features=2)
    pacer.set_scaler(mean=[4.0, 0.0], scale=[1.0, 1.0])

    print(f"🚀 Starting LSTM Hint Server on port {ARGS.port}...")
    app.run(host="0.0.0.0", port=ARGS.port)
