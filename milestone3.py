import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Air Quality Alert System",
    page_icon="🌫️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #2E86AB;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-card {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .day-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #e9ecef;
        margin: 0.2rem;
    }
    .aqi-good { background-color: #00E396; color: white; }
    .aqi-moderate { background-color: #FFA726; color: white; }
    .aqi-unhealthy-sensitive { background-color: #FF6B6B; color: white; }
    .aqi-unhealthy { background-color: #FF4757; color: white; }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">Air Quality Alert System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Milestone 3: Working Application (Weeks 5-6)</div>', unsafe_allow_html=True)

# Main layout columns
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Current Air Quality Section
    st.markdown('<div class="section-header">Current Air Quality</div>', unsafe_allow_html=True)
    
    # Station selector
    station = st.selectbox("Select Station", ["Downtown Station", "Suburban Station", "Industrial Station"])
    
    # Current AQI metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Current AQI", value="145", delta="Moderate")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Primary Pollutant", value="PM2.5", delta="")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Health Impact", value="Moderate", delta="")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Pollutant Concentrations Chart
    st.markdown('<div class="section-header">Pollutant Concentrations</div>', unsafe_allow_html=True)
    
    # Create pollutant data
    pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    concentrations = [45, 38, 52, 28, 12, 0.8]
    
    fig_pollutants = go.Figure(data=[
        go.Bar(
            x=pollutants,
            y=concentrations,
            marker_color=['#FF6B6B', '#FFA726', '#00E396', '#2E86AB', '#A23B72', '#F18F01']
        )
    ])
    
    fig_pollutants.update_layout(
        height=300,
        xaxis_title="Pollutants",
        yaxis_title="Concentration (μg/m³)",
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_pollutants, use_container_width=True)

with col2:
    # 7-Day Forecast Section
    st.markdown('<div class="section-header">7-Day Forecast</div>', unsafe_allow_html=True)
    
    # Forecast data
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    aqi_values = [145, 153, 118, 78, 112, 118, 185]
    aqi_labels = ['Moderate', 'Moderate', 'Unhealthy for Sensitive', 'Moderate', 'Moderate', 'Moderate', 'Unhealthy']
    
    # Display day cards
    for i, (day, aqi, label) in enumerate(zip(days, aqi_values, aqi_labels)):
        # Determine AQI color class
        if aqi <= 50:
            aqi_class = "aqi-good"
        elif aqi <= 100:
            aqi_class = "aqi-moderate"
        elif aqi <= 150:
            aqi_class = "aqi-unhealthy-sensitive"
        else:
            aqi_class = "aqi-unhealthy"
        
        st.markdown(f"""
        <div class="day-card {aqi_class}">
            <strong>{day}</strong><br>
            AQI {aqi}<br>
            <small>{label}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # AQI Color Legend
    st.markdown("---")
    st.markdown("**AQI Color Scale:**")
    legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
    
    with legend_col1:
        st.markdown('<div style="background-color: #00E396; color: white; padding: 5px; border-radius: 5px; text-align: center; font-size: 0.8rem;">Good</div>', unsafe_allow_html=True)
    with legend_col2:
        st.markdown('<div style="background-color: #FFA726; color: white; padding: 5px; border-radius: 5px; text-align: center; font-size: 0.8rem;">Moderate</div>', unsafe_allow_html=True)
    with legend_col3:
        st.markdown('<div style="background-color: #FF6B6B; color: white; padding: 5px; border-radius: 5px; text-align: center; font-size: 0.8rem;">Unhealthy for Sensitive</div>', unsafe_allow_html=True)
    with legend_col4:
        st.markdown('<div style="background-color: #FF4757; color: white; padding: 5px; border-radius: 5px; text-align: center; font-size: 0.8rem;">Unhealthy</div>', unsafe_allow_html=True)

with col3:
    # Active Alerts Section
    st.markdown('<div class="section-header">Active Alerts</div>', unsafe_allow_html=True)
    
    alerts = [
        {
            "level": "Unhealthy for Sensitive Groups",
            "message": "High pollution levels expected",
            "date": "Thursday, 25/07/16"
        },
        {
            "level": "High Ozone Levels Expected",
            "message": "Ozone concentration rising",
            "date": "Friday, 25/07/16"
        },
        {
            "level": "Moderate Air Quality",
            "message": "Standard precautions advised",
            "date": "Friday, 8/09/04"
        }
    ]
    
    for alert in alerts:
        # Determine alert color based on level
        if "Unhealthy" in alert["level"]:
            border_color = "#FF4757"
        elif "High" in alert["level"]:
            border_color = "#FF6B6B"
        else:
            border_color = "#FFA726"
        
        st.markdown(f"""
        <div class="alert-card" style="border-left-color: {border_color};">
            <div style="font-weight: bold; color: {border_color}; margin-bottom: 0.5rem;">
                {alert['level']}
            </div>
            <div style="margin-bottom: 0.5rem;">
                {alert['message']}
            </div>
            <div style="font-size: 0.8rem; color: #666;">
                {alert['date']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Metrics
    st.markdown("---")
    st.markdown('<div class="section-header">Additional Metrics</div>', unsafe_allow_html=True)
    
    additional_metrics = {
        "Temperature": "24°C",
        "Humidity": "65%",
        "Wind Speed": "12 km/h",
        "Pressure": "1013 hPa"
    }
    
    for metric, value in additional_metrics.items():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text(metric)
        with col2:
            st.text(value)
        st.progress(0)  # Placeholder for visual indicator

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Data updates every 15 minutes | Last updated: {}
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

# Hidden expander for raw data (optional)
with st.expander("Raw Data Preview"):
    # Sample data table
    data = {
        'Date': pd.date_range(start='2024-01-01', periods=7, freq='D'),
        'PM2.5': [45, 52, 38, 42, 48, 35, 55],
        'PM10': [38, 45, 32, 38, 42, 30, 48],
        'O3': [52, 58, 45, 50, 55, 42, 60],
        'AQI': [145, 153, 118, 78, 112, 118, 185]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)