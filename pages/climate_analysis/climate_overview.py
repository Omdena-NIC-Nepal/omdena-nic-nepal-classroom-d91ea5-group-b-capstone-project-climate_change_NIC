import streamlit as st
from scripts.data_utils import load_data, remove_unwanted_columns, clean_data

def display():
    st.subheader("Data Preview")

    # Load and preprocess the dataset if not already loaded
    if "cleaned_climatedf" not in st.session_state:
        climatedf = load_data("data/climate_data/climate_data_nepal_district_wise_monthly.csv")
        cleaned_climatedf = clean_data(climatedf, method='dropna')
        st.session_state.cleaned_climatedf = cleaned_climatedf

    cleaned_climatedf = st.session_state.cleaned_climatedf
    st.dataframe(cleaned_climatedf, height=200)

    st.divider()

    st.markdown("""
    This dataset captures **monthly climate statistics** across various districts in **Nepal**, with measurements recorded from **1981 onwards**. Below is a description of each feature:

    ### 📅 Temporal Columns
    - **DATE**: Specific day of data recording (e.g., "1/31/1981")
    - **YEAR, MONTH**: Extracted from `DATE`, useful for trend analysis over time

    ### 📍 Geographical Data
    - **DISTRICT**: Name of the district where data was recorded
    - **LAT, LON**: Latitude and longitude coordinates of the district location

    ### 🌧️ Precipitation and Pressure
    - **PRECTOT**: Total monthly precipitation in millimeters
    - **PS**: Surface atmospheric pressure in kilopascals (kPa)

    ### 💧 Humidity and Moisture
    - **QV2M**: Specific humidity at 2 meters (grams of water vapor per kg of air)
    - **RH2M**: Relative humidity at 2 meters (percentage)

    ### 🌡️ Temperature Measures
    - **T2M**: Mean air temperature at 2 meters
    - **T2MWET**: Wet bulb temperature at 2 meters
    - **T2M_MAX / T2M_MIN**: Max and Min air temperature at 2 meters
    - **T2M_RANGE**: Temperature range (max - min)
    - **TS**: Surface skin temperature

    ### 🌬️ Wind Speed Metrics
    - **WS10M / WS50M**: Mean wind speed at 10m and 50m height
    - **WS10M_MAX / WS50M_MAX**: Max wind speed at 10m and 50m
    - **WS10M_MIN / WS50M_MIN**: Min wind speed at 10m and 50m
    - **WS10M_RANGE / WS50M_RANGE**: Wind speed range at both heights

    ---

    This structured meteorological dataset supports comprehensive climate research and modeling, particularly:
    - Identifying climate change patterns
    - Studying extreme weather events
    - Performing district-level comparative climate analysis

    Use the interactive features to visualize and explore how climate factors vary across space and time in Nepal. 🌿
    """)

    st.divider()
