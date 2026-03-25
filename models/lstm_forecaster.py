import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMForecaster:

    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.model = LSTMModel()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def prepare_data(self, data):
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length])

        return np.array(X), np.array(y)

    def train(self, data, epochs=5):
        X, y = self.prepare_data(data)

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)

        for epoch in range(epochs):
            self.model.train()

            outputs = self.model(X)
            loss = self.criterion(outputs.squeeze(), y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def predict(self, data):
        self.model.eval()

        X, _ = self.prepare_data(data)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            preds = self.model(X).squeeze().numpy()

        return preds