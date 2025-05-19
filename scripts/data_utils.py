import pandas as pd
import streamlit as st
import geopandas as gpd
import folium
@st.cache_data
@st.cache_data
def load_data(file: str, **kwargs) -> pd.DataFrame | None:
    """
    Load data from a CSV file into a pandas DataFrame.
    Accepts additional keyword arguments (e.g., skiprows).
    """
    try:
        df = pd.read_csv(file, encoding='utf-8', **kwargs)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data from {file}: {e}")
        return None


def remove_unwanted_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """
    Remove unwanted columns from the DataFrame.
    """
    return df.drop(columns=columns_to_remove, errors='ignore')

def clean_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Clean the DataFrame using the specified method.
    """
    if method == 'dropna':
        return df.dropna()
    elif method == 'fillna':
        return df.fillna(0)
    elif method == 'interpolate':
        return df.interpolate()
    else:
        raise ValueError("Invalid cleaning method. Use 'dropna', 'fillna', or 'interpolate'.")

