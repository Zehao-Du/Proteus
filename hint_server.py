#!/usr/bin/env python3
import math
import pandas as pd
import time
import sys
import os
import joblib
from flask import Flask, jsonify

app = Flask(__name__)

# Models
ISO_MODEL_PATH = "isolation_forest.pkl"
GBDT_MODEL_PATH = "gbdt_model.pkl"
DATA_PATH = "net_data.csv"

class HealthPredictor:
    def __init__(self):
        self.iso_bundle = None
        self.gbdt_bundle = None
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(ISO_MODEL_PATH):
                self.iso_bundle = joblib.load(ISO_MODEL_PATH)
            if os.path.exists(GBDT_MODEL_PATH):
                self.gbdt_bundle = joblib.load(GBDT_MODEL_PATH)
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

predictor = HealthPredictor()

def s_curve(x, k=5):
    return 1 / (1 + math.exp(-k * (x - 0.5)))

def pace_from_health(h):
    base_rate = 5.0
    max_rate = 100.0
    factor = s_curve(h)
    return round(base_rate + factor * (max_rate - base_rate), 2)

@app.route("/hint", methods=["GET"])
def get_hint():
    try:
        # Reload models periodically or just once? Kept simple here.
        # predictor.load_models() 
        
        # Read latest data
        # We need the last row to have all features including rolling ones
        # smart_agent.py writes rolling features, so we can just take the last row
        df = pd.read_csv(DATA_PATH).tail(5)
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

if __name__ == "__main__":
    print("🚀 Starting Hint Server on port 5000...")
    app.run(host="0.0.0.0", port=5000)

