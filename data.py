import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

# ================== LOAD ENV VARIABLES ==================
load_dotenv()  # loads variables from .env file
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # fallback default

# ================== CONFIG ==================
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "f315af223cca48b5a6167d7490e7280a650df750")

CITY_FEED = {
    "Delhi": "new delhi",
    "Mumbai": "mumbai",
    "Kolkata": "kolkata",
    "Chennai": "chennai"
}

LSTM_MODEL_PATH = "pm25_lstm_model.keras"

# Load model and historical dataset
lstm_model = load_model(LSTM_MODEL_PATH)
df_hist = pd.read_csv('Data/Real-Data/Real_Combine.csv').dropna()

# ================== FEATURES ==================
all_columns = df_hist.columns.tolist()
TARGET_COL = next((c for c in all_columns if "PM" in c and "2.5" in c), None)
FEATURE_COLS = [c for c in all_columns if c not in [TARGET_COL] and df_hist[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(df_hist[FEATURE_COLS].values)
scaler_y.fit(df_hist[[TARGET_COL]].values)

TIME_STEPS = 3
POLLUTANT_MAP = {pol: pol for pol in all_columns if pol.replace(".", "") in all_columns or pol in all_columns}

# ================== HELPER FUNCTIONS ==================
def fetch_city_aqi(city):
    feed = CITY_FEED.get(city)
    if not feed:
        return None
    url = f"https://api.waqi.info/feed/{feed}/?token={AQICN_TOKEN}"
    try:
        data = requests.get(url, timeout=10).json()
        if data.get("status") != "ok":
            return None
        d = data["data"]
        iaqi = d.get("iaqi", {})
        pollutants = {k.upper(): iaqi.get(k.lower(), {}).get("v") for k in ["pm25","pm10","no2","o3","so2","co"]}
        return {"city": city, "aqi": d.get("aqi"), "dominentpol": d.get("dominentpol"), "pollutants": pollutants, "time": d.get("time", {}).get("iso")}
    except:
        return None

def calculate_aqi_category(aqi):
    if aqi is None:
        return "Unknown", "#808080", "❓"
    aqi = float(aqi)
    if aqi <= 50: return "Good", "#00E396", "😊"
    elif aqi <= 100: return "Satisfactory", "#A3E4D7", "🙂"
    elif aqi <= 200: return "Moderate", "#FFA726", "😐"
    elif aqi <= 300: return "Poor", "#FF6B6B", "😷"
    elif aqi <= 400: return "Very Poor", "#FF4757", "😨"
    else: return "Severe", "#8B0000", "🚨"

def get_health_advice(aqi_category):
    advice = {
        "Good": "Air quality is good. 🟢 Enjoy outdoor activities.",
        "Satisfactory": "Some pollutants may be a bit elevated; sensitive people should take precaution.",
        "Moderate": "Acceptable, but sensitive groups may see effects. Limit heavy outdoor exertion.",
        "Poor": "Health effects possible for all; sensitive groups avoid outdoor activities.",
        "Very Poor": "Health alert! Everyone may be affected. Avoid outdoor activity.",
        "Severe": "Emergency! Stay indoors; reduce all exposure."
    }
    return advice.get(aqi_category, "No advice.")

def predict_next_pm25():
    latest_features = df_hist[FEATURE_COLS].iloc[-TIME_STEPS:].values
    if latest_features.shape[1] != lstm_model.input_shape[2]:
        st.warning("Feature dimension mismatch with LSTM model. Cannot predict PM2.5.")
        return None
    X_seq = latest_features.reshape(1, TIME_STEPS, len(FEATURE_COLS))
    y_pred_scaled = lstm_model.predict(X_seq, verbose=0)
    next_pm25 = scaler_y.inverse_transform(y_pred_scaled)[0][0]
    return next_pm25

# ================== STREAMLIT LAYOUT ==================
st.set_page_config(page_title="AirAware – Real-time AQI + LSTM", layout="wide")
st.title("AirAware – Real-time Air Quality Monitoring with LSTM Prediction")
st.write("Data from AQICN → for Delhi, Mumbai, Kolkata, Chennai")

# ----------------- ADMIN LOGIN -----------------
st.sidebar.title("Admin Login")
admin_password_input = st.sidebar.text_input("Enter Admin Password", type="password")
admin_access = admin_password_input == ADMIN_PASSWORD

if admin_access:
    st.sidebar.success("Admin Access Granted")
    st.sidebar.subheader("🔹 Admin Panel")

    # Upload dataset
    uploaded_file = st.sidebar.file_uploader("Upload new CSV dataset", type="csv")
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")

        if st.sidebar.button("Retrain LSTM Model"):
            with st.spinner("Retraining model..."):
                # Scale new data
                scaler_X.fit(df_new[FEATURE_COLS].values)
                scaler_y.fit(df_new[[TARGET_COL]].values)

                # Prepare sequences
                X_seq, y_seq = [], []
                for i in range(TIME_STEPS, len(df_new)):
                    X_seq.append(df_new[FEATURE_COLS].iloc[i-TIME_STEPS:i].values)
                    y_seq.append(df_new[TARGET_COL].iloc[i])
                X_seq, y_seq = np.array(X_seq), np.array(y_seq).reshape(-1,1)
                y_seq_scaled = scaler_y.transform(y_seq)
                X_seq_scaled = scaler_X.transform(X_seq.reshape(-1, len(FEATURE_COLS))).reshape(X_seq.shape)

                # Retrain
                lstm_model.fit(X_seq_scaled, y_seq_scaled, epochs=5, batch_size=32, verbose=0)
                lstm_model.save(LSTM_MODEL_PATH)
                st.sidebar.success("LSTM Model retrained and saved!")

    # PM2.5 Alert threshold
    alert_thresh = st.sidebar.number_input("Set PM2.5 Alert Threshold", min_value=0, max_value=500, value=200)
else:
    alert_thresh = 200

# ----------------- MAIN APP -----------------
city = st.selectbox("Select a city", list(CITY_FEED.keys()))
if city:
    with st.spinner("Fetching live data..."):
        info = fetch_city_aqi(city)

    if info:
        aqi = info["aqi"]
        aqi_cat, aqi_color, aqi_emoji = calculate_aqi_category(aqi)
        st.subheader(f"{city} — AQI: {aqi} {aqi_emoji} ({aqi_cat})")
        st.write(f"Dominant pollutant: {info.get('dominentpol')}")

        if aqi and float(aqi) > alert_thresh:
            st.warning(f"⚠️ Current AQI high: {aqi} — Take precautions!")

        poll = info["pollutants"]
        cols = st.columns(3)
        for i, (pol, val) in enumerate(poll.items()):
            with cols[i % 3]:
                st.metric(pol, val if val is not None else "—")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi,
            domain={"x": [0,1], "y": [0,1]},
            title={"text": f"AQI — {aqi_cat}"},
            gauge={"axis":{"range":[0,500]},"bar":{"color": aqi_color}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.info(get_health_advice(aqi_cat))

        st.markdown("### 🔮 Predicted PM2.5 for Next Time Step")
        next_pm25 = predict_next_pm25()
        if next_pm25 is not None:
            st.metric("Predicted PM2.5", round(next_pm25,2))
            if next_pm25 > alert_thresh:
                st.error(f"Forecast PM2.5 High: {round(next_pm25,2)} — Take precautions!")

        # Historical trends
        st.markdown("### 📊 Historical Trends (PM2.5 & Pollutants)")
        if TARGET_COL in df_hist.columns:
            fig_line = px.line(df_hist, x='T', y=TARGET_COL, title=f'{city} — PM2.5 Historical Trend', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

# Footer
st.markdown("---")
st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local)")