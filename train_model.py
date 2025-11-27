#!/usr/bin/env python3
"""Train an Isolation Forest on collected network telemetry with richer reporting."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "avg_rtt_us",
    "p95_rtt_us",
    "retrans_count",
    "rolling_avg_rtt_us",
    "rolling_p95_rtt_us",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Isolation Forest for SmartNetDiag")
    parser.add_argument("--data", default="net_data.csv", help="input CSV path")
    parser.add_argument("--model", default="isolation_forest.pkl", help="output model path")
    parser.add_argument("--contamination", type=float, default=0.2, help="expected anomaly ratio")
    parser.add_argument("--estimators", type=int, default=150, help="number of trees")
    parser.add_argument("--test-size", type=float, default=0.2, help="hold-out ratio for quick validation")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES, help="feature columns to train on")
    return parser.parse_args()


def load_data(path: str, features):
    df = pd.read_csv(path)
    df = df.dropna(subset=features)
    X = df[features]
    return df, X


def train_model(X, contamination: float, n_estimators: int, random_state: int = 42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler, X_scaled


def evaluate(model, scaler, df: pd.DataFrame, features, test_size: float):
    _, X_test = train_test_split(df[features], test_size=test_size, random_state=42, shuffle=True)
    X_test_scaled = scaler.transform(X_test)

    df_test = df.loc[X_test.index].copy()
    df_test["prediction"] = model.predict(X_test_scaled)
    df_test["score"] = model.decision_function(X_test_scaled)

    # Assume labels are optional; if absent, skip confusion matrix
    metrics = {}
    if "label" in df.columns:
        y_true = df_test["label"].apply(lambda x: -1 if str(x).lower() == "anomaly" or str(x) == "-1" else 1)
        y_pred = df_test["prediction"]
        cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
        metrics["confusion_matrix"] = cm.tolist()
        metrics["normal_precision"] = cm[0][0] / max(cm[0].sum(), 1)
        metrics["anomaly_precision"] = cm[1][1] / max(cm[1].sum(), 1)
    return df_test, metrics


def plot_results(df: pd.DataFrame, features, out_path: Path):
    plt.figure(figsize=(10, 6))
    colors = df["prediction"].map({1: "green", -1: "red"})
    plt.scatter(df[features[0]], df["retrans_count"], c=colors, alpha=0.6, marker="o")
    plt.xlabel(features[0])
    plt.ylabel("retrans_count")
    plt.title("Isolation Forest predictions (green=normal, red=anomaly)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)


def save_artifacts(model, scaler, model_path: str):
    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, model_path)


def main():
    args = parse_args()
    df, X = load_data(args.data, args.features)
    print(f"Loaded {len(df)} samples from {args.data} with features {args.features}")

    model, scaler, X_scaled = train_model(X, args.contamination, args.estimators)

    df_pred = df.copy()
    df_pred["prediction"] = model.predict(X_scaled)
    df_pred["score"] = model.decision_function(X_scaled)

    # Quick validation split for feedback
    df_test, metrics = evaluate(model, scaler, df, args.features, args.test_size)

    save_artifacts(model, scaler, args.model)
    print(f"✅ Saved model + scaler to {args.model}")

    result_plot = Path("model_result.png")
    plot_results(df_pred, args.features, result_plot)
    print(f"✅ Saved visualization to {result_plot}")

    metrics_path = Path("training_metrics.json")
    metrics_payload = {
        "samples": len(df),
        "features": args.features,
        "contamination": args.contamination,
        "estimators": args.estimators,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"✅ Wrote training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
