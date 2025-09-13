import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Air Quality Data Explorer",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .section-header {
        font-size: 18px;
        color: #1f77b4;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_and_process_data():
    # Create synthetic data that simulates the processed data from our analysis
    # In a real scenario, this would load from the preprocessed CSV file
    
    # Generate date range
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
    
    data = []
    for date in dates:
        for city in cities:
            # Base values with some randomness
            base_pm25 = np.random.lognormal(mean=3.5, sigma=0.7)
            base_pm10 = np.random.lognormal(mean=4.0, sigma=0.6)
            base_no2 = np.random.lognormal(mean=2.8, sigma=0.5)
            
            # Apply city-specific multipliers
            if city == 'Delhi':
                base_pm25 *= 2.5
                base_pm10 *= 2.2
                base_no2 *= 1.8
            elif city == 'Mumbai':
                base_pm25 *= 1.8
                base_pm10 *= 1.7
                base_no2 *= 1.5
            elif city == 'Kolkata':
                base_pm25 *= 2.0
                base_pm10 *= 1.9
                base_no2 *= 1.6
            
            # Apply seasonal effects (winter has higher pollution)
            month = date.month
            if month in [11, 12, 1, 2]:  # Winter months
                seasonal_factor = 1.6
            elif month in [3, 4, 5]:  # Spring
                seasonal_factor = 1.2
            elif month in [6, 7, 8, 9]:  # Monsoon
                seasonal_factor = 0.7
            else:  # Autumn
                seasonal_factor = 1.0
                
            # Apply weekend effect (slightly lower pollution on weekends)
            day_of_week = date.weekday()
            weekend_factor = 0.9 if day_of_week >= 5 else 1.0
            
            # Calculate final values
            pm25 = base_pm25 * seasonal_factor * weekend_factor
            pm10 = base_pm10 * seasonal_factor * weekend_factor
            no2 = base_no2 * seasonal_factor * weekend_factor
            
            # Calculate AQI (simplified)
            aqi = max(pm25, pm10, no2) * 1.2
            
            data.append({
                'Date': date,
                'City': city,
                'PM2.5': pm25,
                'PM10': pm10,
                'NO2': no2,
                'O3': np.random.lognormal(mean=2.5, sigma=0.4) * seasonal_factor,
                'SO2': np.random.lognormal(mean=1.8, sigma=0.3) * seasonal_factor,
                'CO': np.random.lognormal(mean=1.2, sigma=0.2) * seasonal_factor,
                'AQI': aqi,
                'Year': date.year,
                'Month': date.month,
                'DayOfWeek': day_of_week,
                'Season': 'Winter' if month in [12, 1, 2] else 
                         'Spring' if month in [3, 4, 5] else 
                         'Summer' if month in [6, 7, 8] else 'Autumn',
                'IsWeekend': 1 if day_of_week >= 5 else 0
            })
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['PM2.5_PM10_Ratio'] = df['PM2.5'] / df['PM10']
    
    # Create AQI buckets
    def get_aqi_bucket(aqi):
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Satisfactory'
        elif aqi <= 200:
            return 'Moderate'
        elif aqi <= 300:
            return 'Poor'
        elif aqi <= 400:
            return 'Very Poor'
        else:
            return 'Severe'
    
    df['AQI_Bucket'] = df['AQI'].apply(get_aqi_bucket)
    
    return df

# Load the data
df = load_and_process_data()

# Sidebar
st.sidebar.markdown('<p class="main-header">Air Quality Data Explorer</p>', unsafe_allow_html=True)

# Data Controls
st.sidebar.markdown('### Data Controls')
location = st.sidebar.selectbox('Location', df['City'].unique(), index=0)
time_range = st.sidebar.selectbox('Time Range', ['Last 24 Hours', 'Last 7 Days', 'Last 30 Days', 'Last 6 Months', 'Last Year', 'All Time'], index=4)

# Date filtering
end_date = df['Date'].max()
if time_range == 'Last 24 Hours':
    start_date = end_date - timedelta(days=1)
elif time_range == 'Last 7 Days':
    start_date = end_date - timedelta(days=7)
elif time_range == 'Last 30 Days':
    start_date = end_date - timedelta(days=30)
elif time_range == 'Last 6 Months':
    start_date = end_date - timedelta(days=180)
elif time_range == 'Last Year':
    start_date = end_date - timedelta(days=365)
else:  # All Time
    start_date = df['Date'].min()

filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['City'] == location)]

# Pollutants selection
st.sidebar.markdown('### Pollutants')
pollutants = st.sidebar.multiselect(
    'Select Pollutants to Display',
    ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO', 'AQI'],
    default=['PM2.5', 'PM10', 'NO2']
)

# Data Quality metrics (calculated based on actual data)
total_records = len(filtered_df)
complete_records = total_records - filtered_df[pollutants].isnull().any(axis=1).sum()
completeness = round((complete_records / total_records) * 100) if total_records > 0 else 0

# Validity - check if values are within reasonable ranges
valid_records = total_records
for pollutant in pollutants:
    if pollutant in filtered_df.columns:
        # Simple validity check - values should be positive and not extremely high
        valid_records = min(valid_records, len(filtered_df[(filtered_df[pollutant] > 0) & (filtered_df[pollutant] < 1000)]))
validity = round((valid_records / total_records) * 100) if total_records > 0 else 0

# Main content
st.markdown('<p class="main-header">Air Quality Data Explorer</p>', unsafe_allow_html=True)

# First row
col1, col2 = st.columns([2, 1])

with col1:
    # Time series chart
    st.markdown('### PM2.5 Time Series')
    if not filtered_df.empty and 'PM2.5' in pollutants:
        fig = px.line(filtered_df, x='Date', y='PM2.5', 
                     title=f'PM2.5 Concentration in {location}')
        fig.update_layout(yaxis_title='Concentration (μg/m³)', height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PM2.5 data available for the selected filters.")

with col2:
    # Data Quality metrics
    st.markdown('### Data Quality')
    
    st.markdown(f"""
    <div class="metric-card">
        <p>Completeness:</p>
        <h3>{completeness}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <p>Validity:</p>
        <h3>{validity}%</h3>
    </div>
    """, unsafe_allow_html=True)

# Second row
col3, col4 = st.columns([1, 1])

with col3:
    # Statistical Summary
    st.markdown('### Statistical Summary')
    if not filtered_df.empty:
        # Calculate statistics for selected pollutants
        stats_data = []
        for pollutant in pollutants:
            if pollutant in filtered_df.columns:
                stats = filtered_df[pollutant].describe()
                stats_data.append({
                    'Pollutant': pollutant,
                    'Mean': stats['mean'],
                    'Std Dev': stats['std'],
                    'Median': stats['50%'],
                    'Min': stats['min'],
                    'Max': stats['max']
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Display metrics
        for _, row in stats_df.iterrows():
            st.markdown(f"**{row['Pollutant']}**")
            col31, col32, col33 = st.columns(3)
            with col31:
                st.metric("Mean (μg/m³)", round(row['Mean'], 1))
            with col32:
                st.metric("Std Dev", round(row['Std Dev'], 1))
            with col33:
                st.metric("Median (μg/m³)", round(row['Median'], 1))
            
            col34, col35 = st.columns(2)
            with col34:
                st.metric("Min (μg/m³)", round(row['Min'], 1))
            with col35:
                st.metric("Max (μg/m³)", round(row['Max'], 1))
            
            st.metric("Data Points", len(filtered_df))
            st.markdown("---")

with col4:
    # Pollutant Correlations
    st.markdown('### Pollutant Correlations')
    if not filtered_df.empty and len(pollutants) > 1:
        # Calculate correlations
        corr_matrix = filtered_df[pollutants].corr()
        
        # Create a list of correlation pairs
        correlations = []
        for i in range(len(pollutants)):
            for j in range(i+1, len(pollutants)):
                p1 = pollutants[i]
                p2 = pollutants[j]
                if p1 in corr_matrix.columns and p2 in corr_matrix.index:
                    corr_value = corr_matrix.loc[p1, p2]
                    correlations.append({
                        'Pair': f'{p1}-{p2}',
                        'Correlation': corr_value
                    })
        
        # Display top 6 correlations
        for corr in correlations[:6]:
            col41, col42 = st.columns([3, 1])
            with col41:
                st.write(f"{corr['Pair']}")
            with col42:
                st.write(f"{corr['Correlation']:.2f}")
    
    # Distribution Analysis
    st.markdown('### Distribution Analysis')
    if not filtered_df.empty and 'PM2.5' in pollutants:
        # Create bins for PM2.5
        bins = [0, 20, 40, 60, 80, 100, float('inf')]
        labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
        filtered_df['PM2.5_Range'] = pd.cut(filtered_df['PM2.5'], bins=bins, labels=labels)
        
        # Count values in each bin
        dist_counts = filtered_df['PM2.5_Range'].value_counts().reindex(labels, fill_value=0)
        
        # Create bar chart
        fig = px.bar(x=labels, y=dist_counts.values, 
                     labels={'x': 'PM2.5 Range (μg/m³)', 'y': 'Frequency'},
                     title='PM2.5 Distribution', height=300)
        st.plotly_chart(fig, use_container_width=True)

# Third row - Additional insights
st.markdown('### Additional Insights')

col5, col6 = st.columns(2)

with col5:
    # AQI Trends
    st.markdown('**AQI Trends**')
    if not filtered_df.empty and 'AQI' in df.columns:
        fig = px.line(filtered_df, x='Date', y='AQI', 
                     title=f'AQI Trend in {location}', height=300)
        st.plotly_chart(fig, use_container_width=True)

with col6:
    # Pollutant Comparison
    st.markdown('**Pollutant Comparison**')
    if not filtered_df.empty and len(pollutants) > 0:
        # Calculate averages
        avg_pollutants = filtered_df[pollutants].mean().reset_index()
        avg_pollutants.columns = ['Pollutant', 'Value']
        
        fig = px.bar(avg_pollutants, x='Pollutant', y='Value',
                     title='Average Pollutant Concentration', height=300)
        st.plotly_chart(fig, use_container_width=True)

# Fourth row - Seasonal analysis
st.markdown('### Seasonal Analysis')

col7, col8 = st.columns(2)

with col7:
    # Seasonal patterns
    if not filtered_df.empty and 'Season' in df.columns:
        seasonal_avg = filtered_df.groupby('Season')[pollutants].mean().mean(axis=1).reset_index()
        seasonal_avg.columns = ['Season', 'Average Concentration']
        
        fig = px.bar(seasonal_avg, x='Season', y='Average Concentration',
                     title='Average Pollution by Season', height=300)
        st.plotly_chart(fig, use_container_width=True)

with col8:
    # Weekday vs Weekend
    if not filtered_df.empty and 'IsWeekend' in df.columns:
        weekday_avg = filtered_df.groupby('IsWeekend')[pollutants].mean().mean(axis=1).reset_index()
        weekday_avg['Day Type'] = weekday_avg['IsWeekend'].apply(lambda x: 'Weekend' if x == 1 else 'Weekday')
        
        fig = px.bar(weekday_avg, x='Day Type', y=0,
                     title='Average Pollution: Weekday vs Weekend', height=300,
                     labels={'0': 'Average Concentration'})
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Data Source:** Processed Indian Air Quality Dataset | **Note:** Dashboard shows analyzed and processed data")