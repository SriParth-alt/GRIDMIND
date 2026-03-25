import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

# 🔥 IMPORTANT: filter one building
df = df[df["building_id"] == 0]

# 🔥 REMOVE ZERO VALUES (MAIN FIX)
df = df[df["meter_reading"] > 0]

df = df.sort_values("timestamp")

demand = df["meter_reading"].values


# -----------------------
# NORMALIZATION
# -----------------------
scaler = MinMaxScaler()
demand_scaled = scaler.fit_transform(demand.reshape(-1, 1)).flatten()


# -----------------------
# TRAIN LSTM
# -----------------------
model = LSTMForecaster(sequence_length=24)
model.train(demand_scaled, epochs=15)   # slightly increased


# -----------------------
# SAVE
# -----------------------
torch.save(model.model.state_dict(), "lstm_model.pth")
joblib.dump(scaler, "lstm_scaler.save")

print("LSTM model + scaler saved")