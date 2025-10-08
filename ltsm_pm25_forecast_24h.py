# ltsm_pm25_forecast_24h.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# ================== Step 1: Load Data ==================
df = pd.read_csv("Data/Real-Data/Real_Combine.csv")  # Fix path slashes
df['PM 2.5'] = df['PM 2.5'].interpolate(method='linear')
df = df.fillna(method='bfill')

features = ['T','TM','Tm','SLP','H','VV','V','VM']
target = ['PM 2.5']

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[target])

# ================== Step 2: Create Sequences ==================
def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i+time_steps)])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# ================== Step 3: Train/Test Split ==================
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# ================== Step 4: Build & Train LSTM ==================
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# ================== Step 5: Predict Next 24 Hours ==================
last_sequence = X_scaled[-time_steps:]  # last 24 readings, shape (24, 8)
current_seq = last_sequence.copy()
predictions_scaled = []

for _ in range(24):  # predict next 24 hours
    # Predict next PM2.5
    pred_scaled = model.predict(current_seq.reshape(1, time_steps, len(features)), verbose=0)
    predictions_scaled.append(pred_scaled[0, 0])

    # Build next feature vector (shape 8,)
    next_features = np.zeros(len(features))
    next_features[0] = pred_scaled[0, 0]  # put predicted PM2.5 into first slot (T can be placeholder)

    # Shift window and append new row
    current_seq = np.vstack([current_seq[1:], next_features])

# Inverse transform predictions
predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1,1))

# ================== Step 6: Save Predictions ==================
future_hours = pd.date_range(start=pd.Timestamp.now(), periods=24, freq='H')
pred_df = pd.DataFrame({
    'Hour': future_hours,
    'Predicted_PM25': predictions.flatten()
})
pred_df.to_csv("pm25_next_24h.csv", index=False)
print("Next 24h PM2.5 predictions saved to pm25_next_24h.csv")

# ================== Step 7: Plot ==================
plt.figure(figsize=(12,6))
plt.plot(range(len(df)), df['PM 2.5'], label='Historical PM2.5')
plt.plot(range(len(df), len(df)+24), predictions, label='Next 24h Prediction', color='red')
plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.legend()
plt.show()
