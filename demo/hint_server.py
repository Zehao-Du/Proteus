#!/usr/bin/env python3
import math
import pandas as pd
import time
import sys
import os
import joblib
import argparse
from flask import Flask, jsonify

app = Flask(__name__)

# Global variables placeholder
predictor = None
ARGS = None  # To store parsed arguments

class HealthPredictor:
    def __init__(self, iso_path, gbdt_path):
        self.iso_path = iso_path
        self.gbdt_path = gbdt_path
        self.iso_bundle = None
        self.gbdt_bundle = None
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(self.iso_path):
                self.iso_bundle = joblib.load(self.iso_path)
                print(f"✅ Loaded Isolation Forest from {self.iso_path}")
            else:
                print(f"⚠️ Model not found: {self.iso_path}")

            if os.path.exists(self.gbdt_path):
                self.gbdt_bundle = joblib.load(self.gbdt_path)
                print(f"✅ Loaded GBDT from {self.gbdt_path}")
            else:
                print(f"⚠️ Model not found: {self.gbdt_path}")
                
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")

    def predict(self, feature_row):
        # feature_row is a DataFrame with 1 row and correct columns
        if not self.iso_bundle or not self.gbdt_bundle:
            # Fallback
            rtt = feature_row['avg_rtt_us'].iloc[0]
            return 1.0 if rtt < 20000 else 0.5

        scaler = self.iso_bundle['scaler'] # Assume same scaler for both
        iso = self.iso_bundle['model']
        gbdt = self.gbdt_bundle['model']

        X_scaled = scaler.transform(feature_row)
        
        # 1. Anomaly Score
        # decision_function: < 0 is anomaly
        anomaly_score = iso.decision_function(X_scaled)[0]
        
        # 2. Predicted Next RTT
        pred_rtt = gbdt.predict(X_scaled)[0]
        
        # 3. Calculate Health
        # RTT factor: 10ms -> 1.0, 100ms -> low
        rtt_factor = 1.0 / (1.0 + max(0, pred_rtt) / 50000.0)
        
        # Anomaly penalty
        penalty = 0.5 if anomaly_score < 0 else 0.0
        
        health = rtt_factor - penalty
        return max(0.01, min(1.0, health))

def s_curve(x, k=5):
    return 1 / (1 + math.exp(-k * (x - 0.5)))

def pace_from_health(h):
    base_rate = 5.0
    max_rate = 100.0
    factor = s_curve(h)
    return round(base_rate + factor * (max_rate - base_rate), 2)

@app.route("/hint", methods=["GET"])
def get_hint():
    global predictor, ARGS
    try:
        if predictor is None:
             return jsonify({"error": "Server initializing"}), 503

        # Read latest data from the configured path
        # smart_agent.py writes rolling features, so we can just take the last row
        if not os.path.exists(ARGS.data_path):
             return jsonify({"health": 1.0, "token_rate": 50.0, "status": "waiting_for_data"})

        df = pd.read_csv(ARGS.data_path).tail(5)
        if df.empty:
            return jsonify({"health": 1.0, "token_rate": 50.0, "status": "no_data"})
            
        latest = df.iloc[[-1]] # Keep as DataFrame
        
        feature_cols = ['avg_rtt_us', 'p95_rtt_us', 'retrans_count', 'rolling_avg_rtt_us', 'rolling_p95_rtt_us']
        latest_features = latest[feature_cols]
        
        health = predictor.predict(latest_features)
        rate = pace_from_health(health)
        
        return jsonify({
            "health": health,
            "token_rate": rate,
            "metrics": {
                "rtt": int(latest['avg_rtt_us'].iloc[0]),
                "retrans": int(latest['retrans_count'].iloc[0])
            }
        })
    except Exception as e:
        return jsonify({"health": 0.5, "token_rate": 10.0, "error": str(e)})

def parse_args():
    parser = argparse.ArgumentParser(description="AI Hint Server")
    parser.add_argument("--iso-model", type=str, default="isolation_forest.pkl", help="Path to Isolation Forest model")
    parser.add_argument("--gbdt-model", type=str, default="gbdt_model.pkl", help="Path to GBDT model")
    parser.add_argument("--data-path", type=str, default="net_data.csv", help="Path to network data CSV")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    return parser.parse_args()

if __name__ == "__main__":
    ARGS = parse_args()
    
    print(f"🔧 Configuration:")
    print(f"   ISO Model:  {ARGS.iso_model}")
    print(f"   GBDT Model: {ARGS.gbdt_model}")
    print(f"   Data Path:  {ARGS.data_path}")

    # Initialize predictor with parsed paths
    predictor = HealthPredictor(ARGS.iso_model, ARGS.gbdt_model)

    print(f"🚀 Starting Hint Server on port {ARGS.port}...")
    app.run(host="0.0.0.0", port=ARGS.port)