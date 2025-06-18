import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px 

from pathlib import Path
from matplotlib.animation import FuncAnimation

DATA_PATH = Path(__file__).resolve().parents[1] / "Weather_&_Climate_Data" / "Processed Datas" /"Feature_Engineering"/ "feature_engineered_climate_data_scaled.csv"

if not DATA_PATH.exists():
    st.error(f"File not found: {DATA_PATH}")
    st.stop()
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['Interval_Start'])
df = df.sort_values('date')

st.title("📊 Weather Data Visualization Dashboard")

# Sidebar Filter
districts = df['District'].dropna().unique().tolist()
selected_district = st.sidebar.selectbox("Select District", ['All'] + districts)
if selected_district != 'All':
    df = df[df['District'] == selected_district]

# Time-Series Plot
st.subheader("📈 Temperature and Precipitation Over Time")
fig1, ax1 = plt.subplots(figsize = (12,6))
ax1.plot(df['Interval_Start'], df['Temp_2m'], label='Temperature',color ='red')
ax1.set_ylabel('Temperature (°C)', color='red')
ax2 = ax1.twinx()
ax2.plot(df['Interval_Start'], df['Precip'], label='Precipitation', color='blue')
ax2.set_ylabel('Precipitation (mm)', color='blue')
ax1.set_title('Temperature and Precipitation Time Series')
st.pyplot(fig1)

# Heatmap of Weather Metrics
st.subheader("🌡️ Heatmap of Average Weather Metrics Across Districts")
heat_df = df.groupby('District')[['Temp_2m', 'Precip', 'Heat_Stress_Index', 'avg_windspeed']].mean()
fig2, ax2 = plt.subplots(figsize=(12,8))
sns.heatmap(heat_df, annot=True, cmap='coolwarm', ax=ax2, annot_kws={'size' : 8})
ax2.set_title("Heatmap of Climate Metrics by District")
st.pyplot(fig2)

# Geospatial Map with Weather Anomalies
st.subheader("🗺️ Geospatial Weather Anomalies Map")
df['Extreme_Event'] = (
    (df['Temp_2m'] >= np.percentile(df['Temp_2m'], 95)) & 
    (df['Humidity_2m'] < np.percentile(df['Humidity_2m'], 5)) & 
    (df['WindSpeed_10m'] > np.percentile(df['WindSpeed_10m'], 95))
).astype(int)
fig3 = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Drought_Index",
                         hover_data=["Temp_2m", "Precip"],
                         mapbox_style="open-street-map", zoom=4, title="Drought Anomalies Map")
st.plotly_chart(fig3)

# Animated Change Over Time
st.subheader("🌍 Animated Monthly Weather Changes")

# Data preparation
df['month_year'] = pd.to_datetime(df['Interval_Start']).dt.strftime('%Y-%m')
monthly_avg = df.groupby(['month_year', 'District'], as_index=False)[['Temp_2m']].mean()

# Create figure with persistent markers
fig = px.line(
    monthly_avg,
    x='month_year',
    y='Temp_2m',
    color='District',
    title="Monthly Average Temperature Over Time",
    animation_frame='month_year',
    range_y=[monthly_avg['Temp_2m'].min()-2, monthly_avg['Temp_2m'].max()+2],
    labels={'Temp_2m': 'Temperature (°C)'},
    height=600
)

fig.update_traces(
    mode='lines+markers',  # Show both lines AND markers
    marker=dict(size=8),   # Explicit marker size
    line=dict(width=2)     # Line thickness
)

# Proper animation configuration
fig.update_layout(
    xaxis_title='Month-Year',
    yaxis_title='Temperature (°C)',
    xaxis={'tickangle': 45},
    showlegend=True,
    legend_title_text='Districts'
)

# Correct animation settings
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300

st.plotly_chart(fig, use_container_width=True)

# Boxplot of Monthly SPI/HSI
st.subheader("📦 Monthly Distribution of SPI and HSI")
df['month'] = pd.to_datetime(df['Interval_Start']).dt.month
fig6, ax6 = plt.subplots(figsize=(12, 5))
sns.boxplot(x='month', y='Drought_Index', data=df, ax=ax6)
ax6.set_title('Monthly Distribution of SPI-like (Drought Index)')
st.pyplot(fig6)

fig7, ax7 = plt.subplots(figsize=(12, 5))
sns.boxplot(x='month', y='Heat_Stress_Index', data=df, ax=ax7)
ax7.set_title('Monthly Distribution of Heat Stress Index')
st.pyplot(fig7)

# scatter_mapboxap Plot for Lag Relationship
st.subheader("🔁 Lag Relationship: Precipitation vs Disaster Occurrence")
if 'Extreme_Event' in df.columns:
    fig8 = px.scatter(df[df['Extreme_Event'].notna()], x='precip_lag1', y='temp_2m_lag1',
                      color='Extreme_Event', title='Precipitation Lag 7 vs Temperature Lag 7 at Disaster Times')
    st.plotly_chart(fig8)

# Correlation Matrix
st.subheader("🔗 Climate Feature Correlation Matrix")
features = ['Temp_2m', 'Precip', 'Humidity_2m','avg_windspeed', 'Drought_Index', 'Heat_Stress_Index']
fig11, ax11 = plt.subplots()
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax11)
ax11.set_title("Correlation Matrix of Climate Features")
st.pyplot(fig11)

# Show full data
if st.checkbox("Show Full Data Table"):
    st.write(df)