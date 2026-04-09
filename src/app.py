# app.py
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import numpy as np

app = Flask(__name__, static_folder="static")
CORS(app)

CSV_PATH = os.path.join("models", "combined_results_lstm_ppo.csv")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/api/results")
def results():
    if not os.path.exists(CSV_PATH):
        return jsonify({
            "error": f"CSV not found at {CSV_PATH}. Make sure combined_results_lstm_ppo.csv is inside src/models/."
        }), 404

    df = pd.read_csv(CSV_PATH)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/summary")
def summary():
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": f"CSV not found at {CSV_PATH}"}), 404

    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "episode",
        "step",
        "actual_glucose",
        "predicted_glucose",
        "action",
        "reward",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns in CSV: {missing}"}), 400

    actual = df["actual_glucose"].astype(float).values
    pred = df["predicted_glucose"].astype(float).values

    denom = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - np.sum((actual - pred) ** 2) / denom if denom != 0 else 0

    summary_data = {
        "time_in_range": round(float(np.mean((actual >= 70) & (actual <= 180)) * 100), 2),
        "hypo_events": int(np.sum(actual < 70)),
        "hyper_events": int(np.sum(actual > 180)),
        "mae": round(float(np.mean(np.abs(actual - pred))), 2),
        "rmse": round(float(np.sqrt(np.mean((actual - pred) ** 2))), 2),
        "r2": round(float(r2), 4),
        "avg_reward": round(float(df["reward"].mean()), 2),
        "total_steps": int(len(df)),
        "num_episodes": int(df["episode"].nunique()),
    }

    return jsonify(summary_data)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)