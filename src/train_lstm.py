import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib


SEQ_LEN = 6        # how many past timesteps to look at
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 0.001
EPOCHS = 5

# Load data
df = pd.read_csv("data/processed/patient_data.csv", parse_dates=["timestamp"])
df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

FEATURES = ["glucose_level", "basal", "bolus", "meal_carbs", "exercise_intensity"]
TARGET = "glucose_level"

df[FEATURES] = df[FEATURES].fillna(0)

# Scale features
scaler = MinMaxScaler()
scaler.fit(df[FEATURES].values)
df[FEATURES] = scaler.transform(df[FEATURES].values)
joblib.dump(scaler, "models/scaler.save")
print("Scaler saved.")


#Patient-wise train/test split
def split_patients(df, train_ratio=0.8):
    patient_ids = df['patient_id'].unique()
    n_train = int(len(patient_ids) * train_ratio)
    train_ids = patient_ids[:n_train]
    test_ids = patient_ids[n_train:]
    train_df = df[df['patient_id'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df

train_df, test_df = split_patients(df, train_ratio=0.8)

class GlucoseDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        self.sequences = []
        self.targets = []

        for pid, patient_df in df.groupby("patient_id"):
            data = np.array(patient_df[FEATURES].values, dtype=np.float32)
            target = np.array(patient_df[TARGET].values, dtype=np.float32)
            for i in range(len(data) - seq_len):
                self.sequences.append(data[i:i+seq_len])
                self.targets.append(target[i+seq_len])

        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

train_dataset = GlucoseDataset(train_df)
test_dataset = GlucoseDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# LSTM Model 
class GlucoseLSTM(nn.Module):
    def __init__(self, input_size=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = GlucoseLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for seq, target in train_loader:
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss/len(train_dataset):.6f}")

 
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/glucose_lstm.pth")
print("Model saved to models/glucose_lstm.pth")

# Evaluation 
model.eval()
all_targets = []
all_preds = []

with torch.no_grad():
    for seq, target in test_loader:
        pred = model(seq)
        all_targets.extend(target.numpy().flatten())
        all_preds.extend(pred.numpy().flatten())

mse = mean_squared_error(all_targets, all_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)
accuracy_percent = r2 * 100

print(f"\nTest Set Metrics")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R2 Score: {r2:.4f}")
print(f"Accuracy of the LSTM model: {accuracy_percent:.2f}%")

#  Plot for one test patient 
patient_id = test_df['patient_id'].unique()[0]
patient_df = test_df[test_df['patient_id'] == patient_id].reset_index(drop=True)

sequences = []
targets = []
data = patient_df[FEATURES].values
target_vals = patient_df[TARGET].values
for i in range(len(data) - SEQ_LEN):
    sequences.append(data[i:i+SEQ_LEN])
    targets.append(target_vals[i+SEQ_LEN])

sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
targets = np.array(targets)

with torch.no_grad():
    predictions = model(sequences).numpy().flatten()

plt.figure(figsize=(12,5))
plt.plot(patient_df['timestamp'][SEQ_LEN:], targets, label='Actual')
plt.plot(patient_df['timestamp'][SEQ_LEN:], predictions, label='Predicted')
plt.title(f'Glucose Prediction for Patient {patient_id} (Test Set)')
plt.xlabel('Time')
plt.ylabel('Glucose Level (scaled)')
plt.legend()
plt.show()