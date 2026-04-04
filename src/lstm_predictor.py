import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

SEQ_LEN = 6
FEATURES = ["glucose_level", "basal", "bolus", "meal_carbs", "exercise_intensity"]
HIDDEN_SIZE = 64
NUM_LAYERS = 2


class GlucoseLSTM(nn.Module):
    def __init__(self, input_size=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTMPredictor:
    """
    Wraps the trained GlucoseLSTM model.
    Maintains a rolling buffer of the last SEQ_LEN observations,
    and predicts the next glucose level (in mg/dL).
    """

    def __init__(self, model_path: str, data_csv: str):
        # fit scaler on processed data so we can scale/inverse-scale
        df = pd.read_csv(data_csv)
        df[FEATURES] = df[FEATURES].fillna(0)
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[FEATURES].values)

        # model loading
        self.model = GlucoseLSTM()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # rolling buffer: list of raw (unscaled) feature vectors
        self.buffer = []

    def reset(self):
        """Clear the buffer at the start of a new episode."""
        self.buffer = []

    def update(self, glucose: float, basal: float, bolus: float,
               meal_carbs: float, exercise_intensity: float):
        """
        Push the latest observation into the buffer.
        Keeps only the last SEQ_LEN entries.
        """
        obs = [glucose, basal, bolus, meal_carbs, exercise_intensity]
        self.buffer.append(obs)
        if len(self.buffer) > SEQ_LEN:
            self.buffer.pop(0)

    def predict(self) -> float:
        """
        Returns predicted next glucose (mg/dL), or None if buffer not full yet.
        """
        if len(self.buffer) < SEQ_LEN:
            return None

        raw = np.array(self.buffer, dtype=np.float32)          # (SEQ_LEN, 5)
        scaled = self.scaler.transform(raw)                     # (SEQ_LEN, 5)
        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, SEQ_LEN, 5)

        with torch.no_grad():
            pred_scaled = self.model(tensor).item()             # scaled glucose

        # inverse-transform: only glucose column [0]
        dummy = np.zeros((1, len(FEATURES)), dtype=np.float32)
        dummy[0, 0] = pred_scaled
        pred_mg_dl = self.scaler.inverse_transform(dummy)[0, 0]
        return float(pred_mg_dl)