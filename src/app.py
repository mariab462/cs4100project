# app.py

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import os
import numpy as np

app = Flask(__name__, static_folder="static")
CORS(app)

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
models_dir = os.path.join(project_root, "models")

csv_path = os.path.join(models_dir, "combined_results_lstm_ppo.csv")
lstm_path = os.path.join(models_dir, "glucose_lstm.pth")
ppo_policy_path = os.path.join(models_dir, "ppo_policy_lstm.pth")

_models = {}


def get_models():
    if _models:
        return _models

    import torch
    from train_lstm import GlucoseLSTM
    from train_ppo import PolicyNetwork
    from env.project_env import ProjectEnv

    env = ProjectEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    lstm = GlucoseLSTM(input_size=5, hidden_size=64, num_layers=2)
    lstm.load_state_dict(torch.load(lstm_path, map_location="cpu"))
    lstm.eval()

    policy = PolicyNetwork(obs_dim, action_dim)
    policy.load_state_dict(torch.load(ppo_policy_path, map_location="cpu"))
    policy.eval()

    _models["lstm"] = lstm
    _models["policy"] = policy
    _models["env"] = env
    return _models


def pad_to_six(arr, fill_value=0.0):
    arr = list(arr)
    if len(arr) >= 6:
        return np.array(arr[-6:], dtype=np.float32)
    if len(arr) == 0:
        arr = [fill_value]
    while len(arr) < 6:
        arr.insert(0, arr[0])
    return np.array(arr, dtype=np.float32)


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def scale_glucose(x):
    return clip01((x - 40.0) / (400.0 - 40.0))


def unscale_glucose(x):
    return float(x) * (400.0 - 40.0) + 40.0


def scale_basal(x):
    return clip01(x / 5.0)


def scale_bolus(x):
    return clip01(x / 10.0)


def scale_meals(x):
    return clip01(x / 100.0)


def scale_activity(x):
    return clip01(x / 3.0)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/api/results")
def results():
    if not os.path.exists(csv_path):
        return jsonify({"error": f"csv not found at {csv_path}"}), 404
    df = pd.read_csv(csv_path)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/summary")
def summary():
    if not os.path.exists(csv_path):
        return jsonify({"error": f"csv not found at {csv_path}"}), 404

    df = pd.read_csv(csv_path)
    required_cols = ["episode", "step", "actual_glucose", "predicted_glucose", "action", "reward"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return jsonify({"error": f"missing columns: {missing}"}), 400

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

    cgm = pad_to_six(data.get("cgm", [120] * 6), fill_value=120.0)
    meals = pad_to_six(data.get("meals", [0] * 6), fill_value=0.0)
    insulin = pad_to_six(data.get("insulin", [0] * 6), fill_value=0.0)
    activity = pad_to_six(data.get("activity", [0] * 6), fill_value=0.0)

    models = get_models()
    lstm = models["lstm"]
    policy = models["policy"]
    env = models["env"]

    basal = np.zeros(6, dtype=np.float32)

    seq = np.column_stack([
        scale_glucose(cgm),
        scale_basal(basal),
        scale_bolus(insulin),
        scale_meals(meals),
        scale_activity(activity)
    ]).astype(np.float32)

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        lstm_out = lstm(seq_tensor)
        predicted_glucose_scaled = float(lstm_out.item())

    predicted_glucose = round(float(np.clip(unscale_glucose(predicted_glucose_scaled), 40, 400)), 1)

    current_glucose = float(cgm[-1])
    prev_glucose = float(cgm[-2])
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
    scaled_mu_per_min = ((raw_action + 1.0) / 2.0) * (env_high - env_low) + env_low
    ppo_insulin_dose = max(0.0, round(scaled_mu_per_min / 1000.0 * 5.0, 3))

    fallback_dose = 0.0
    if predicted_glucose >= 190 and ppo_insulin_dose < 0.1:
        fallback_dose = min(1.0, round((predicted_glucose - 180.0) / 60.0, 3))

        if float(activity[-1]) > 0:
            fallback_dose = max(0.0, round(fallback_dose - 0.2, 3))

        if float(insulin[-1]) > 0:
            fallback_dose = max(0.0, round(fallback_dose - 0.2, 3))

    if fallback_dose > 0:
        insulin_dose = fallback_dose
        dose_source = "fallback"
    else:
        insulin_dose = ppo_insulin_dose
        dose_source = "ppo"

    if predicted_glucose < 70:
        status = "low"
        status_text = "low — hold insulin"
    elif predicted_glucose > 180 and insulin_dose > 0:
        status = "high"
        if dose_source == "fallback":
            status_text = f"high — {insulin_dose} units suggested"
        else:
            status_text = f"high — {insulin_dose} units suggested"
    elif predicted_glucose > 180:
        status = "elevated"
        status_text = "elevated — no insulin recommended"
    else:
        status = "in_range"
        status_text = "in target range (70–180)"

    return jsonify({
        "predicted_glucose": predicted_glucose,
        "insulin_dose": insulin_dose,
        "status": status,
        "status_text": status_text,
        "dose_source": dose_source,
        "used_lstm": True,
        "used_ppo": True,
        "raw_ppo_action": round(raw_action, 4),
        "seq_len_used": 6
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)