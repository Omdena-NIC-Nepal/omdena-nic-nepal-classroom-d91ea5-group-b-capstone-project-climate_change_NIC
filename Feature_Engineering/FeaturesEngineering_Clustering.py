import requests
import pandas as pd
import time
import folium
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic

def get_elevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/srtm90m?locations={lat},{lon}"
    response = requests.get(url)
    time.sleep(1)
    data = response.json()
    return data['results'][0]['elevation']
    
df = pd.read_csv("../Weather_&_Climate_Data/Processed Datas/5yeargrouped.csv")
# Apply to your dataframe
df['Elevation'] = df.apply(lambda row: get_elevation(row['Latitude'], row['Longitude']), axis=1)
#df.to_csv("../Weather & Climate Data/Processed Datas/Feature_Engineering/Climate_data_with_Elevation.csv")


def process_station_data(raw_df):
    """Process the station data and ensure proper numeric coordinates"""
    # Extract latitude and longitude rows
    lat_row = raw_df[raw_df['Indicators'] == 'Latitude'].iloc[0]
    lon_row = raw_df[raw_df['Indicators'] == 'Longitude'].iloc[0]
    
    # Create station dataframe
    stations = []
    for col in raw_df.columns[1:]:  # Skip 'Indicators Months' column
        try:
            lat = float(lat_row[col])
            lon = float(lon_row[col])
            stations.append({
                'StationName': col,
                'Latitude': lat,
                'Longitude': lon
            })
        except (ValueError, TypeError):
            continue  # Skip invalid entries
    
    return pd.DataFrame(stations)

def find_nearest_station(lat, lon, stations_df):
    """Find closest station with proper numeric validation"""
    min_dist = float('inf')
    nearest = None
    
    for _, station in stations_df.iterrows():
        try:
            station_lat = float(station['Latitude'])
            station_lon = float(station['Longitude'])
            dist = geodesic((lat, lon), (station_lat, station_lon)).km
            if dist < min_dist:
                min_dist = dist
                nearest = station['StationName']
        except (ValueError, TypeError):
            continue
            
    return nearest

def create_map(district_df, station_raw_df):
    """Create the visualization map with error handling"""
    stations_df = process_station_data(station_raw_df)
    
    # Verify we have valid stations
    if stations_df.empty:
        raise ValueError("No valid stations found with proper coordinates")
    
    # Assign nearest stations with validation
    district_df['NearestStation'] = district_df.apply(
        lambda r: find_nearest_station(r['Latitude'], r['Longitude'], stations_df),
        axis=1
    )
    
    # Create map centered on Nepal
    m = folium.Map(location=[27.7172, 85.3240], zoom_start=7)
    
    # Add stations with validation
    for _, s in stations_df.iterrows():
        try:
            folium.Marker(
                [float(s['Latitude']), float(s['Longitude'])],
                popup=f"Station: {s['StationName']}",
                icon=folium.Icon(color='red', icon='cloud')
            ).add_to(m)
        except (ValueError, TypeError):
            continue
    
    # Add districts with color coding
    valid_districts = district_df.dropna(subset=['NearestStation'])
    colors = {s: f"#{random.randint(0,0xFFFFFF):06x}" 
             for s in valid_districts['NearestStation'].unique()}
    
    for _, d in valid_districts.iterrows():
        try:
            folium.CircleMarker(
                [float(d['Latitude']), float(d['Longitude'])],
                radius=6,
                color=colors[d['NearestStation']],
                fill=True,
                popup=f"{d['District']} → {d['NearestStation']}"
            ).add_to(m)
        except (ValueError, TypeError):
            continue
    
    return m

raw_station_data = pd.read_csv('../Weather & Climate Data/Weather Station Data.csv')  # Your first table
district_data = pd.read_csv("../Weather & Climate Data/Processed Datas/grouped_5yr.csv")    # Your second table

weather_map = create_map(district_data, raw_station_data)
#weather_map.save('nepal_weather_map.html')




