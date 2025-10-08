import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_TOKEN = "f315af223cca48b5a6167d7490e7280a650df750"
LSTM_MODEL_PATH = "pm25_lstm_model.keras"
SCALER_PATH = os.getenv("SCALER_PATH")

# Load trained LSTM model
lstm_model = load_model(LSTM_MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

st.title("AirAware 🌤️ – Real-time AQI + LSTM Prediction")
st.markdown("Interactive dashboard for Delhi, Mumbai, Kolkata, Chennai using live AQICN data and PM2.5 LSTM forecasts.")

# City selection
cities = ["Delhi", "Mumbai", "Kolkata", "Chennai"]
selected_city = st.selectbox("Select a city 🌆", cities)

# Function to fetch real-time AQI from CPCB API
def fetch_aqi(city):
    url = f"https://api.cpcb.gov.in/aqi?city={city}&token={API_TOKEN}"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return data
    else:
        st.error("Failed to fetch data from CPCB API")
        return None

# Fetch AQI data
aqi_data = fetch_aqi(selected_city)
if aqi_data:
    pm25 = aqi_data.get('pm25', 0)
    o3 = aqi_data.get('o3', 0)
    pm10 = aqi_data.get('pm10', 0)
    so2 = aqi_data.get('so2', 0)
    dominant_pollutant = aqi_data.get('dominant_pollutant', 'pm25')
    st.metric(f"{selected_city} — AQI", pm25, delta=None)
    st.write(f"Dominant pollutant: {dominant_pollutant}")
    
    # LSTM prediction
    last_24h_pm25 = np.array(aqi_data.get('pm25_last_24h', [pm25]*24))
    last_24h_pm25_scaled = scaler.transform(last_24h_pm25.reshape(-1,1))
    X_input = last_24h_pm25_scaled.reshape(1,24,1)
    pred_scaled = lstm_model.predict(X_input)
    pred_pm25 = scaler.inverse_transform(pred_scaled)
    st.write(f"Predicted next hour PM2.5: {pred_pm25[0][0]:.2f}")
    
    # Plot last 24h PM2.5
    hours = [f"{i}h ago" for i in range(23,-1,-1)]
    df_plot = pd.DataFrame({'Hour': hours, 'PM2.5': last_24h_pm25})
    fig = px.line(df_plot, x='Hour', y='PM2.5', title=f"{selected_city} PM2.5 last 24h")
    st.plotly_chart(fig)
