import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from models.lstm_forecaster import LSTMForecaster


# -----------------------
# LOAD DATA
# -----------------------
base_path = "dataset/"

train = pd.read_csv(base_path + "train.csv")
building_metadata = pd.read_csv(base_path + "building_metadata.csv")
weather = pd.read_csv(base_path + "weather_train.csv")

df = train.merge(building_metadata, on="building_id", how="left")
df = df.merge(weather, on=["site_id", "timestamp"], how="left")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df[df["building_id"] == 0]

# 🔥 REMOVE ZERO VALUES
df = df[df["meter_reading"] > 0]

df = df.sort_values("timestamp")

demand = df["meter_reading"].values


# -----------------------
# LOAD SCALER
# -----------------------
scaler = joblib.load("lstm_scaler.save")

demand_scaled = scaler.transform(demand.reshape(-1, 1)).flatten()


# -----------------------
# SEQUENCES
# -----------------------
def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)


# scaled for input
X, _ = create_sequences(demand_scaled)

# original for actual
_, y_actual = create_sequences(demand)


X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)


# -----------------------
# LOAD MODEL
# -----------------------
model = LSTMForecaster(sequence_length=24)
model.model.load_state_dict(torch.load("lstm_model.pth"))
model.model.eval()

with torch.no_grad():
    preds = model.model(X).squeeze().numpy()


# -----------------------
# INVERSE SCALE
# -----------------------
preds = scaler.inverse_transform(preds.reshape(-1, 1))


# -----------------------
# DEBUG PRINT
# -----------------------
print("Actual sample:", y_actual[:10])
print("Pred sample:", preds[:10])


# -----------------------
# PLOT
# -----------------------
plt.figure()
plt.plot(y_actual[:500], label="Actual")
plt.plot(preds[:500], label="Predicted")
plt.legend()
plt.title("LSTM Forecast vs Actual (ASHRAE)")

plt.show()