import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import geopandas.geodataframe as gdf
from IPython.display import display
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandera as pa
from pandera import Column, DataFrameSchema

import pandas as pd
from datetime import datetime, timedelta

def scale_to_quin():
    # Load the dataset with optimized parameters
    datasset = pd.read_csv(
        "../Weather_&_Climate_Data/Historical temperature, precipitation, and weather events/raw/climate_data/dailyclimate.csv",
        parse_dates=['Date'],  # Parse dates during read
        dtype={'District': 'category'}  # Use categorical for district to save memory
    )
    datasset = datasset.loc[:, ~datasset.columns.duplicated()]

    # Define starting date and interval size
    start_date = pd.to_datetime('1981-01-01')
    interval_years = 4
    interval_months = 2
    interval_days = 10

    # Vectorized interval calculation
    def calculate_intervals(dates):
        # Convert to numpy datetime64[ns] for faster operations
        dates_np = dates.values.astype('datetime64[ns]')
        start_np = start_date.to_datetime64()
        
        # Calculate total days since start_date
        deltas = (dates_np - start_np).astype('timedelta64[D]').astype(int)
        
        # Calculate total days in one interval
        interval_days_total = (interval_years * 365.25 + 
                             interval_months * 30.44 + 
                             interval_days)
        
        # Calculate interval number for each date
        interval_num = (deltas / interval_days_total).astype(int)
        
        # Calculate start date of each interval
        interval_starts = start_np + (interval_num * interval_days_total).astype('timedelta64[D]')
        
        return pd.to_datetime(interval_starts)

    # Apply vectorized interval calculation
    datasset['Interval_Start'] = calculate_intervals(datasset['Date'])

    # Group by District and Interval_Start with optimized aggregation
    grouped_5yr = datasset.groupby(
        ['District', 'Interval_Start'], 
        observed=True  # For categorical grouping optimization
    ).mean(numeric_only=True).reset_index()
    
    # Remove unnamed columns more efficiently
    grouped_5yr = grouped_5yr.loc[:, ~grouped_5yr.columns.str.contains('^Unnamed')]
    
    # Extract year and month
    grouped_5yr['Year'] = grouped_5yr['Interval_Start'].dt.year
    grouped_5yr['Month'] = grouped_5yr['Interval_Start'].dt.month
    
    # Sort values
    grouped_5yr = grouped_5yr.sort_values(['District', 'Year', 'Month'])

    # Identify columns to fill
    cols_to_fill = grouped_5yr.columns.difference(
        ['Interval_Start', 'Year', 'Month', 'District', 'Latitude', 'Longitude']
    )

    # Vectorized filling of missing values using shift
    for col in cols_to_fill:
        # Create a temporary DataFrame with shifted values
        temp = grouped_5yr[['District', 'Year', 'Month', col]].copy()
        temp['Year'] -= 1  # Previous year
        temp = temp.rename(columns={col: 'prev_value'})
        
        # Merge once for all columns (outside the loop if possible)
        grouped_5yr = grouped_5yr.merge(
            temp, 
            on=['District', 'Year', 'Month'], 
            how='left'
        )
        
        # Fill missing values
        grouped_5yr[col] = grouped_5yr[col].fillna(grouped_5yr['prev_value'])
        grouped_5yr = grouped_5yr.drop(columns=['prev_value'])

    # Save the result
    grouped_5yr.to_csv(
        "../Weather_&_Climate_Data/Processed Datas/5yeargrouped.csv", 
        index=False
    )

scale_to_quin()
        

def temporal_alignment():
        df = pd.read_csv(r"../Weather_&_Climate_Data/Processed Datas/5yeargrouped.csv")
        date_range = pd.date_range(start=df['Interval_Start'].min(), end=df['Interval_Start'].max(), freq='ME')
        districts = df['District'].unique()
        full_index = pd.MultiIndex.from_product([districts, date_range], names=['DISTRICT', 'DATE'])
        df['DATE'] = pd.to_datetime(df['Interval_Start'], format = "mixed", dayfirst = False)
        df = df.set_index(['District', 'DATE']).reindex(full_index).reset_index()
        print(df.head())
temporal_alignment()

def georeferenced_dataset():
    # Load data with optimized parameters
    nepal_boundaries = gpd.read_file(r"../Weather_&_Climate_Data/Local Unit/local_unit.shp", usecols=["Province", "DISTRICT", "geometry"])
    
    climate_data = pd.read_csv(
        "../Weather_&_Climate_Data/Historical temperature, precipitation, and weather events/raw/climate_data/dailyclimate.csv",
        usecols=["District", "Year", "Temp_2m", "Precip"],  # Only load needed columns
        dtype={'District': 'string', 'Year': 'int16'}  # Optimize dtypes
    )
    
    # Pre-process district names once
    climate_data['District'] = climate_data['District'].str.strip().str.upper()
    nepal_boundaries['DISTRICT'] = nepal_boundaries['DISTRICT'].str.strip().str.upper()

    # Efficient aggregation
    annual_temp = climate_data.groupby(
        ["District", "Year"], 
        observed=True  # Better performance for categoricals
    ).agg(
        Avg_Temp_C=("Temp_2m", "mean"),
        Total_Prec_mm=("Precip", "sum")
    ).reset_index()

    # CRS transformation if needed
    if nepal_boundaries.crs.is_geographic:
        nepal_boundaries = nepal_boundaries.to_crs(epsg=32645)
    
    # Dissolve with categorical optimization
    nepal_boundaries['DISTRICT'] = nepal_boundaries['DISTRICT'].astype('category')
    nepal_boundaries = nepal_boundaries.dissolve(
        by="DISTRICT", 
        as_index=False, 
        aggfunc="first"  # More efficient than default
    )

    # Merge with optimized parameters
    merged = nepal_boundaries.merge(
        annual_temp,
        left_on="DISTRICT",
        right_on="District",
        how="right"  # Keep all climate data points
    )

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=nepal_boundaries.crs  # Maintain consistent CRS
    )

def Consistency_check():
    schema = DataFrameSchema({
        "Temp_2m": Column(float, checks=pa.Check.greater_than(-60)),
        "Precip": Column(float, checks=pa.Check.greater_than_or_equal_to(0))
        })
    climate_data = pd.read_csv(r"../Weather_&_Climate_Data/Historical temperature, precipitation, and weather events/raw/climate_data/dailyclimate.csv")
    schema.validate(climate_data)

Consistency_check()
        







