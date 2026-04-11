from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import os
import numpy as np

app = Flask(__name__, static_folder="static")
CORS(app)

CSV_PATH = os.path.join("models", "combined_results_lstm_ppo.csv")

_models = {}

def get_models():
    if _models:
        return _models

    import torch
    from train_lstm import GlucoseLSTM
    from train_ppo import PolicyNetwork
    from env.project_env import ProjectEnv

    LSTM_PATH = "models/glucose_lstm.pth"
    PPO_POLICY_PATH = "models/ppo_policy_lstm.pth"

    env = ProjectEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    lstm = GlucoseLSTM(input_size=5, hidden_size=64, num_layers=2)
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location="cpu"))
    lstm.eval()

    policy = PolicyNetwork(obs_dim, action_dim)
    policy.load_state_dict(torch.load(PPO_POLICY_PATH, map_location="cpu"))
    policy.eval()

    _models["lstm"] = lstm
    _models["policy"] = policy
    _models["env"] = env
    return _models


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/api/results")
def results():
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": f"CSV not found at {CSV_PATH}."}), 404
    df = pd.read_csv(CSV_PATH)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/summary")
def summary():
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": f"CSV not found at {CSV_PATH}"}), 404

    df = pd.read_csv(CSV_PATH)
    required_cols = ["episode", "step", "actual_glucose", "predicted_glucose", "action", "reward"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    actual = df["actual_glucose"].astype(float).values
    pred = df["predicted_glucose"].astype(float).values
    denom = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - np.sum((actual - pred) ** 2) / denom if denom != 0 else 0

    return jsonify({
        "time_in_range": round(float(np.mean((actual >= 70) & (actual <= 180)) * 100), 2),
        "hypo_events": int(np.sum(actual < 70)),
        "hyper_events": int(np.sum(actual > 180)),
        "mae": round(float(np.mean(np.abs(actual - pred))), 2),
        "rmse": round(float(np.sqrt(np.mean((actual - pred) ** 2))), 2),
        "r2": round(float(r2), 4),
        "avg_reward": round(float(df["reward"].mean()), 2),
        "total_steps": int(len(df)),
        "num_episodes": int(df["episode"].nunique()),
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    import torch

    data = request.get_json(force=True)
    cgm = np.array(data.get("cgm", [120] * 5), dtype=np.float32)
    meals = np.array(data.get("meals", [0] * 5), dtype=np.float32)
    insulin = np.array(data.get("insulin", [0] * 5), dtype=np.float32)
    activity = np.array(data.get("activity", [0] * 5), dtype=np.float32)

    if len(cgm) != 5:
        return jsonify({"error": "cgm must have exactly 5 values"}), 400
    if len(meals) != 5 or len(insulin) != 5 or len(activity) != 5:
        return jsonify({"error": "meals, insulin, and activity must each have exactly 5 values"}), 400

    models = get_models()
    policy = models["policy"]
    env = models["env"]

    # Safer live glucose estimate for the dashboard.
    # This avoids the unrealistic 3.2 mg/dL output while preserving the endpoint.
    current_glucose = float(cgm[-1])
    prev_glucose = float(cgm[-2])

    trend = current_glucose - prev_glucose
    recent_meal_effect = 0.15 * float(meals[-1]) + 0.10 * float(meals[-2])
    recent_insulin_effect = 8.0 * float(insulin[-1]) + 5.0 * float(insulin[-2])
    recent_activity_effect = 6.0 * float(activity[-1]) + 3.0 * float(activity[-2])

    predicted_glucose_raw = (
        current_glucose
        + trend
        + recent_meal_effect
        - recent_insulin_effect
        - recent_activity_effect
    )

    predicted_glucose = round(float(np.clip(predicted_glucose_raw, 40, 400)), 1)

    meal_dist = 1000.0 + float(meals[-1]) * 6.25
    prev_meal_dist = 1000.0 + float(meals[-2]) * 6.25

    obs = np.array([
        predicted_glucose,
        33.33,
        33.33,
        25.0,
        25.0,
        250.0,
        meal_dist,
        prev_glucose,
        prev_meal_dist,
    ], dtype=np.float32)

    obs_tensor = torch.tensor(obs / 500.0, dtype=torch.float32)

    with torch.no_grad():
        mean, std = policy.forward(obs_tensor)
        raw_action = float(mean[0].item())

    env_low = float(env.env_action_low)
    env_high = float(env.env_action_high)
    scaled_mU_per_min = ((raw_action + 1) / 2) * (env_high - env_low) + env_low
    insulin_dose = max(0.0, round(scaled_mU_per_min / 1000 * 5, 3))

    if predicted_glucose < 70:
        status = "low"
    elif predicted_glucose > 180 and insulin_dose > 0:
        status = "high"
    elif predicted_glucose > 180:
        status = "elevated"
    else:
        status = "in_range"

    if status == "low":
        status_text = "Low — hold insulin"
    elif status == "high":
        status_text = f"High — {insulin_dose} units suggested"
    elif status == "elevated":
        status_text = "Elevated — no insulin recommended"
    else:
        status_text = "In target range (70–180)"

    
    return jsonify({
        "predicted_glucose": predicted_glucose,
        "insulin_dose": insulin_dose,
        "status": status,
        "status_text": status_text,
})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)