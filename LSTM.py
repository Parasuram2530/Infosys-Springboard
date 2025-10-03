import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Data/Real-Data/Real_Combine.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Data Preprocessing and Cleaning (following the same approach as RandomForest)
# Check for null values using heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Drop null values (same as RandomForest implementation)
df = df.dropna()
print(f"Dataset shape after dropping null values: {df.shape}")

# Separate features and target
X = df.iloc[:, :-1]  # Independent features
y = df.iloc[:, -1]   # Dependent feature (PM 2.5)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Feature Scaling (important for LSTM)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Scale features
X_scaled = scaler_X.fit_transform(X)
# Scale target
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

print("Feature scaling completed")

# Prepare data for LSTM
# LSTM requires 3D input: [samples, timesteps, features]
# Since we don't have time series data, we'll create sequences

def create_sequences(X, y, time_steps=1):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Choose time steps (you can adjust this)
TIME_STEPS = 3
X_sequences, y_sequences = create_sequences(X_scaled, y_scaled, TIME_STEPS)

print(f"Sequences shape: {X_sequences.shape}")
print(f"Target sequences shape: {y_sequences.shape}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=42, shuffle=False
)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# Build LSTM Model
def create_lstm_model(input_shape):
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Third LSTM layer
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))  # Output layer
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    return model

# Create model
model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
print("LSTM Model Summary:")
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
print("Training LSTM model...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1,
    shuffle=False
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions
y_pred_scaled = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
mse = mean_squared_error(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

print("\n" + "="*50)
print("LSTM MODEL EVALUATION")
print("="*50)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test_original, y_pred, alpha=0.6)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')

plt.subplot(1, 2, 2)
plt.plot(y_test_original, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('PM 2.5')
plt.title('Actual vs Predicted Over Samples')
plt.legend()

plt.tight_layout()
plt.show()

# Residual Analysis
residuals = y_test_original.flatten() - y_pred.flatten()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')

plt.tight_layout()
plt.show()

# Feature Importance Analysis (using permutation importance)
from sklearn.inspection import permutation_importance

# Create a simpler model for feature importance (using last sequence step)
def create_simple_model():
    simple_model = Sequential()
    simple_model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    simple_model.add(Dropout(0.3))
    simple_model.add(Dense(32, activation='relu'))
    simple_model.add(Dropout(0.3))
    simple_model.add(Dense(1))
    
    simple_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return simple_model

# Use the last time step for feature importance
X_last_step = X_scaled[TIME_STEPS:]
y_last_step = y_scaled[TIME_STEPS:]

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_last_step, y_last_step, test_size=0.2, random_state=42
)

simple_model = create_simple_model()
simple_model.fit(X_train_simple, y_train_simple, 
                epochs=50, batch_size=32, verbose=0, 
                validation_split=0.2)

# Calculate permutation importance
perm_importance = permutation_importance(
    simple_model, X_test_simple, y_test_simple,
    n_repeats=10, random_state=42
)

# Plot feature importance
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance for PM 2.5 Prediction')
plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
print(importance_df.tail(5))

# Save the model
model.save('pm25_lstm_model.h5')
print("\nLSTM model saved as 'pm25_lstm_model.h5'")

# Comparison with Random Forest (if available)
print("\n" + "="*50)
print("COMPARISON WITH RANDOM FOREST (from notebook)")
print("="*50)
print("Note: Compare these LSTM results with the Random Forest results")
print("from the original notebook to see which performs better.")