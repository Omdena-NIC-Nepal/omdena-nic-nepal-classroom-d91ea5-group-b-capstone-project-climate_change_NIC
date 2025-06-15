import streamlit as st 
import pandas as pd
from pathlib import Path
import folium

from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

daily_climate_data_path = Path(__file__).resolve().parent.parent.parent/"Weather_&_Climate_Data/Historical temperature, precipitation, and weather events/raw/climate_data/dailyclimate.csv"
df = pd.read_csv(daily_climate_data_path)
st.title("Climate Vulnerability Assessment")
st.write("The various impact of weather based on climate factors")

# temp_max_threshold = df['Temp_2m'].quantile(0.95)
# temp_min_threshold = df['Temp_2m'].quantile(0.05)
#precip_max_threshold = df['Precip'].quantile(0.95)
#precip_min_threshold = df['Precip'].quantile(0.05)
temp_max_threshold = 35
temp_min_threshold = 0

precip_max_threshold = 35
precip_min_threshold = 0.03



Precip_vulnerable_areas = df[(df['Precip'] > precip_max_threshold) | (df['Precip'] < precip_min_threshold)]
Temp_vulnerable_areas = df[(df['Temp_2m'] > temp_max_threshold) | (df['Temp_2m'] < temp_min_threshold)]

st.subheader("Districts prone to high temperature")
st.write(Temp_vulnerable_areas[['Date', 'District', 'Temp_2m']])

st.subheader("Districts Vulnerable to High Precipitation")
st.write(Precip_vulnerable_areas[['Date', 'District', 'Precip']])

st.line_chart(df.groupby('Date')['Temp_2m'].mean(), use_container_width=True)
st.line_chart(df.groupby('Date')['Precip'].mean(), use_container_width=True)

 # Initialize map centered on Nepal
nepal_map = folium.Map(location=[27.7172, 85.3240], zoom_start=7)  # Center of Nepal
 # Plotting red dots for high temperature areas
for _, row in Temp_vulnerable_areas.iterrows():
         folium.CircleMarker(
             location=[row["Latitude"], row["Longitude"]],
             radius=6,
             color="red",
             fill=True,
             fill_color="red",
             fill_opacity=0.6
         ).add_to(nepal_map)
 # Plotting red dots for high precipitation areas
for _, row in Precip_vulnerable_areas.iterrows():
         folium.CircleMarker(
             location=[row["Latitude"], row["Longitude"]],
             radius=6,
             color="blue",
             fill=True,
             fill_color="blue",
             fill_opacity=0.6
         ).add_to(nepal_map)
 # Display map in Streamlit
st.subheader("Climate Vulnerability Map")
st.write("This map shows regions vulnerable to high temperature and high precipitation.")
st.markdown("""
 **Legend**  
 🔴 High Temperature  
 🏜️ Low Precipitation(Drought)
 """)
st.write("This map shows regions vulnerable to high temperature and high precipitation.")
folium_static(nepal_map)