<!-- psudo code -->

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Load & Prepare Data
# ---------------------------
# Assume you have a CSV with columns: Date, Open, High, Low, Close, Volume
df = pd.read_csv("stock_data.csv")

# Keep only OHLCV
data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Scale values between 0 and 1
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Parameters
lookback = 30   # number of past days to look at
features = data.shape[1]  # OHLCV = 5
outputs = 4     # predict [Open, High, Low, Close]

# Create dataset (X = last 30 days, y = next day OHLC)
X, y = [], []
for i in range(len(data) - lookback - 1):
    X.append(data[i:i+lookback])         # past 30 days
    y.append(data[i+lookback, :outputs]) # next day's OHLC
X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ---------------------------
# 2. Define Model
# ---------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # take last time step
        out = self.fc(out)
        return out

model = StockLSTM(input_size=features, hidden_size=64, num_layers=2, output_size=outputs)

# ---------------------------
# 3. Training
# ---------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ---------------------------
# 4. Validation
# ---------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test)
    val_loss = criterion(preds, y_test)
    print("Validation Loss:", val_loss.item())

# ---------------------------
# 5. Save & Load Model
# ---------------------------
torch.save(model.state_dict(), "stock_model.pth")

# Load model later
loaded_model = StockLSTM(input_size=features, hidden_size=64, num_layers=2, output_size=outputs)
loaded_model.load_state_dict(torch.load("stock_model.pth"))
loaded_model.eval()

# ---------------------------
# 6. Inference (Next Day Prediction)
# ---------------------------
with torch.no_grad():
    last_window = torch.tensor(X[-1:], dtype=torch.float32)  # last 30 days
    next_day_pred = loaded_model(last_window).numpy()

# Rescale back to original OHLC
next_day_pred_rescaled = scaler.inverse_transform(
    np.concatenate([next_day_pred, np.zeros((1, features - outputs))], axis=1)
)[:, :outputs]

print("Predicted OHLC for next day:", next_day_pred_rescaled)
