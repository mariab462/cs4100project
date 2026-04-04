# test_lstm_predictor.py
from src.lstm_predictor import LSTMPredictor

predictor = LSTMPredictor(
    model_path="../models/glucose_lstm.pth",
    data_csv="data/processed/patient_data.csv"
)

# Simulate 8 steps of fake observations
fake_steps = [
    (120.0, 0.5, 0.0, 0.0, 0.0),  # glucose, basal, bolus, meal_carbs, exercise
    (118.0, 0.5, 0.0, 0.0, 0.0),
    (115.0, 0.5, 1.3, 45.0, 0.0),
    (130.0, 0.5, 0.0, 0.0, 0.0),
    (145.0, 0.5, 0.0, 0.0, 0.0),
    (160.0, 0.5, 0.0, 0.0, 6.0),
    (155.0, 0.5, 0.0, 0.0, 6.0),
    (148.0, 0.5, 0.0, 0.0, 0.0),
]

for i, (glucose, basal, bolus, meal_carbs, exercise) in enumerate(fake_steps):
    predictor.update(glucose, basal, bolus, meal_carbs, exercise)
    prediction = predictor.predict()
    print(f"Step {i+1}: current glucose={glucose}, predicted next={prediction}")