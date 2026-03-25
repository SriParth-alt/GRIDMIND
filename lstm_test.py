import pandas as pd
import matplotlib.pyplot as plt
from models.lstm_forecaster import LSTMForecaster

# Load your existing synthetic data
from data_generator import generate_microgrid_data

df = generate_microgrid_data()

# Use demand
demand = df["demand"].values.reshape(-1, 1)

# Initialize model
lstm = LSTMForecaster(sequence_length=24)

# Train
lstm.train(demand, epochs=5)

# Predict
preds = lstm.predict(demand)

# Align data
actual = demand[24:]

# Plot
plt.figure()
plt.plot(actual, label="Actual Demand")
plt.plot(preds, label="Predicted Demand")
plt.legend()
plt.title("LSTM Demand Forecasting")
plt.show()