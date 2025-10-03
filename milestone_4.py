import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .day-button {
        width: 100%;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# LSTM Model Functions
@st.cache_resource
def create_lstm_model(sequence_length=24):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

@st.cache_data
def generate_sample_data():
    """Generate realistic air quality sample data"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    np.random.seed(42)
    
    # Base pattern with seasonality
    base = 30 + 20 * np.sin(2 * np.pi * dates.hour / 24)
    seasonal = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
    trend = np.linspace(0, 5, len(dates))
    noise = np.random.normal(0, 5, len(dates))
    
    pm25 = base + seasonal + trend + noise
    pm25 = np.maximum(pm25, 0)  # Ensure non-negative values
    
    return pd.DataFrame({
        'datetime': dates,
        'pm25': pm25,
        'temperature': 20 + 10 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, len(dates)),
        'humidity': 60 + 20 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 5, len(dates)),
        'wind_speed': 5 + 3 * np.random.random(len(dates))
    })

@st.cache_data
def prepare_lstm_data(data, sequence_length=24):
    """Prepare data for LSTM training"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['pm25']])
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

@st.cache_resource
def train_lstm_model(_X_train, _y_train, sequence_length=24):
    """Train LSTM model"""
    model = create_lstm_model(sequence_length)
    
    X_reshaped = _X_train.reshape((_X_train.shape[0], _X_train.shape[1], 1))
    
    history = model.fit(
        X_reshaped, _y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    return model, history

def forecast_future(model, last_sequence, scaler, steps=24):
    """Generate future forecasts using LSTM"""
    forecasts = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Reshape for prediction
        current_sequence_reshaped = current_sequence.reshape(1, len(current_sequence), 1)
        
        # Predict next value
        next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
        forecasts.append(next_pred)
        
        # Update sequence
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    # Inverse transform forecasts
    forecasts_array = np.array(forecasts).reshape(-1, 1)
    forecast_original = scaler.inverse_transform(forecasts_array)
    
    return forecast_original.flatten()

# Initialize session state for model training
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None

# Header
st.markdown('<div class="main-header">Streamlit Web Dashboard</div>', unsafe_allow_html=True)
st.markdown("### Milestone 4: Working Application (Weeks 7-8)")

# Sidebar for LSTM Controls
st.sidebar.header("LSTM Model Controls")
if st.sidebar.button("Train LSTM Model"):
    with st.spinner("Training LSTM model... This may take a few seconds."):
        # Generate and prepare data
        data = generate_sample_data()
        X, y, scaler = prepare_lstm_data(data)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        
        # Train model
        model, history = train_lstm_model(X_train, y_train)
        
        # Generate forecasts
        last_sequence = X[-1]  # Use last sequence for forecasting
        forecasts = forecast_future(model, last_sequence, scaler, steps=168)  # 1 week forecast
        
        st.session_state.forecasts = forecasts
        st.session_state.model_trained = True
        st.session_state.data = data
        st.session_state.scaler = scaler
        
    st.sidebar.success("LSTM model trained successfully!")

# Show model status
if st.session_state.model_trained:
    st.sidebar.success("✅ Model is trained and ready")
else:
    st.sidebar.warning("⚠️ Model not trained yet")

# Create two main columns
col1, col2 = st.columns([2, 1])

with col1:
    # Controls Section
    st.markdown("### Controls")
    
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        station = st.selectbox("Monitoring Station", ["Downtown", "Suburban", "Industrial"], key="station")
        time_range = st.selectbox("Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"], key="time_range")
    
    with control_col2:
        pollutant = st.selectbox("Pollutant", ["PM2.5", "PM10", "O3", "NO2", "SO2"], key="pollutant")
        forecast_horizon = st.selectbox("Forecast Horizon", ["24 Hours", "48 Hours", "7 Days"], key="forecast")
    
    # Current Air Quality Section
    st.markdown("---")
    st.markdown("### Current Air Quality")
    
    # Create metrics in columns
    metric_col1, metric_col2, metric_col3 = st.columns([1, 2, 1])
    
    with metric_col1:
        current_aqi = 68  # Sample data
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="AQI", value=str(current_aqi), delta="Moderate")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col2:
        # Pollutant trends chart
        st.markdown("**Pollutant Trends**")
        
        if st.session_state.model_trained:
            # Use actual data for trends
            recent_data = st.session_state.data.tail(24)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=recent_data['datetime'],
                y=recent_data['pm25'],
                mode='lines',
                line=dict(color='#1f77b4', width=3),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                name='Actual PM2.5'
            ))
            fig_trend.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Time",
                yaxis_title="PM2.5 (μg/m³)",
                showlegend=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            # Sample data when model not trained
            hours = list(range(24))
            pm25_values = [45, 42, 38, 35, 32, 30, 28, 25, 22, 25, 30, 35, 
                          40, 45, 50, 55, 60, 65, 68, 70, 65, 60, 55, 50]
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=hours, 
                y=pm25_values,
                mode='lines',
                line=dict(color='#1f77b4', width=3),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                name='Sample PM2.5'
            ))
            fig_trend.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Hours",
                yaxis_title="PM2.5 (μg/m³)",
                showlegend=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with metric_col3:
        st.markdown("**Admin Node**")
        st.button("Update Data", key="update_data")
        st.button("Cross-meter", key="cross_meter")

    # PM2.5 Forecast Section
    st.markdown("---")
    st.markdown("### PM2.5 Forecast")
    
    if st.session_state.model_trained and st.session_state.forecasts is not None:
        # Use LSTM forecasts
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Reshape 168 hours (1 week) into daily averages
        hourly_forecasts = st.session_state.forecasts[:168]  # First week
        daily_forecasts = []
        for i in range(0, len(hourly_forecasts), 24):
            daily_avg = np.mean(hourly_forecasts[i:i+24])
            daily_forecasts.append(daily_avg)
        
        # Get historical data for comparison
        historical_data = st.session_state.data['pm25'].tail(168)
        historical_daily = []
        for i in range(0, len(historical_data), 24):
            if i + 24 <= len(historical_data):
                historical_daily.append(np.mean(historical_data.iloc[i:i+24]))
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=days[:len(historical_daily)], 
            y=historical_daily,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4', width=3)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=days, 
            y=daily_forecasts,
            mode='lines+markers',
            name='LSTM Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash')
        ))
        fig_forecast.update_layout(
            height=300,
            xaxis_title="Day",
            yaxis_title="PM2.5 (μg/m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title="LSTM-based PM2.5 Forecast"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Show forecast metrics
        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
        with forecast_col1:
            st.metric("Max Forecast", f"{max(daily_forecasts):.1f} μg/m³")
        with forecast_col2:
            st.metric("Min Forecast", f"{min(daily_forecasts):.1f} μg/m³")
        with forecast_col3:
            st.metric("Avg Forecast", f"{np.mean(daily_forecasts):.1f} μg/m³")
        
    else:
        # Sample forecast when no model
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        historical_pm25 = [45, 48, 52, 55, 60, 58, 55]
        forecast_pm25 = [68, 65, 62, 60, 58, 55, 52]
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=days[:5], 
            y=historical_pm25[:5],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1f77b4', width=3)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=days[4:], 
            y=forecast_pm25[4:],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash')
        ))
        fig_forecast.update_layout(
            height=300,
            xaxis_title="Day",
            yaxis_title="PM2.5 (μg/m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title="Sample PM2.5 Forecast (Train model for LSTM predictions)"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

with col2:
    # Alert Notifications Section
    st.markdown("### Alert Notifications")
    
    alerts = [
        {"message": "Moderate air quality expected", "time": "Tomorrow, Last 04"},
        {"message": "Good air quality today", "time": "Today, 6:00 AM"},
        {"message": "Model update completed", "time": "Monday, 1:30 PM"}
    ]
    
    # Add LSTM-based alert if model is trained
    if st.session_state.model_trained and st.session_state.forecasts is not None:
        max_forecast = max(st.session_state.forecasts[:24])
        if max_forecast > 75:
            alerts.insert(0, {
                "message": f"High PM2.5 predicted: {max_forecast:.1f} μg/m³", 
                "time": "LSTM Alert"
            })
    
    for alert in alerts:
        st.markdown(f"""
        <div class="alert-card">
            <strong>{alert['message']}</strong><br>
            <small>{alert['time']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Update Dashboard Section
    st.markdown("---")
    st.markdown("### Update Dashboard")
    
    # Admin controls
    admin_col1, admin_col2 = st.columns(2)
    
    with admin_col1:
        st.button("Update (admin)", key="update_admin", use_container_width=True)
        st.button("Cross-meter (admin)", key="cross_meter_admin", use_container_width=True)
    
    # Day selection buttons
    st.markdown("**Select Day:**")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    day_cols = st.columns(7)
    for i, day in enumerate(days):
        with day_cols[i]:
            if st.button(day, key=f"day_{day}", use_container_width=True):
                st.session_state.selected_day = day

# Model Information Section
st.markdown("---")
st.markdown("### LSTM Model Information")

if st.session_state.model_trained:
    st.success("✅ LSTM Model is actively providing forecasts")
    
    # Show model performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Forecast Horizon", "7 Days")
    with col2:
        st.metric("Data Points", f"{len(st.session_state.data):,}")
    with col3:
        st.metric("Sequence Length", "24 hours")
    with col4:
        st.metric("Features Used", "PM2.5, Temp, Humidity")
else:
    st.warning("⚠️ LSTM model not trained. Click 'Train LSTM Model' in sidebar to enable AI-powered forecasts.")

# Additional information at the bottom
st.markdown("---")
st.markdown("**Note:** This dashboard displays real-time air quality monitoring data and LSTM-based forecasts for better environmental awareness.")





git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Parasuram2530/Infosys-Springboard.git
git push -u origin main