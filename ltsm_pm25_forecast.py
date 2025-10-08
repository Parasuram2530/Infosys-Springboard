# ltsm_pm25_forecast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ================== Step 1: Load Data ==================
df = pd.read_csv("Data\Real-Data\Real_Combine.csv")  # <-- replace with your CSV path
print("Data shape:", df.shape)
print(df.head())

# Fill missing PM2.5 values (linear interpolation)
df['PM 2.5'] = df['PM 2.5'].interpolate(method='linear')
df = df.fillna(method='bfill')  # backfill remaining missing values

# ================== Step 2: Feature Scaling ==================
features = ['T','TM','Tm','SLP','H','VV','V','VM']
target = ['PM 2.5']

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[target])

# ================== Step 3: Create Sequences ==================
def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i+time_steps)])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 24  # last 24 readings to predict next PM2.5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
print("Sequences shape:", X_seq.shape, y_seq.shape)

# ================== Step 4: Train/Test Split ==================
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# ================== Step 5: Build LSTM Model ==================
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# ================== Step 6: Train Model ==================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ================== Step 7: Predict & Evaluate ==================
y_pred = model.predict(X_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

print("MSE:", mean_squared_error(y_test_inv, y_pred_inv))
print("MAE:", mean_absolute_error(y_test_inv, y_pred_inv))

# ================== Step 8: Plot Predictions ==================
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual PM2.5', color='blue')
plt.plot(y_pred_inv, label='Predicted PM2.5', color='red')
plt.title("PM2.5 LSTM Forecast")
plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.legend()
plt.show()
