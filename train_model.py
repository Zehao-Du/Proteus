#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import os
import argparse

def train():
    data_path = "net_data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file {data_path} not found. Run smart_agent.py first.")
        return

    print("🔄 Loading data...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    if df.empty or len(df) < 10:
        print("⚠️ Not enough data to train. Need at least 10 samples.")
        return

    # Feature selection
    # We use columns available in smart_agent.py output
    feature_cols = ['avg_rtt_us', 'p95_rtt_us', 'retrans_count', 'rolling_avg_rtt_us', 'rolling_p95_rtt_us']
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=feature_cols)
    X = df_clean[feature_cols]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🧠 Training Isolation Forest (Anomaly Detection)...")
    iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    
    print("📈 Training GBDT (RTT Prediction)...")
    # Target: Next period's rolling_avg_rtt_us
    y = df_clean["rolling_avg_rtt_us"].shift(-1).fillna(df_clean["rolling_avg_rtt_us"].iloc[-1])
    
    gbdt = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbdt.fit(X_scaled, y)
    
    # Save Isolation Forest for Dashboard (bundle format)
    iso_bundle = {"model": iso, "scaler": scaler}
    joblib.dump(iso_bundle, "isolation_forest.pkl")
    print("✅ Saved isolation_forest.pkl (for Dashboard)")

    # Save GBDT for Pacer
    gbdt_bundle = {"model": gbdt, "scaler": scaler}
    joblib.dump(gbdt_bundle, "gbdt_model.pkl")
    print("✅ Saved gbdt_model.pkl (for Pacer)")

if __name__ == "__main__":
    train()
