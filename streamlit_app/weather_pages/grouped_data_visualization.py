import pandas as pd
import streamlit as st 
from pathlib import Path
from io import StringIO

DATA_PATH = Path(__file__).resolve().parent.parent  # Go up 2 directories
interval_data = pd.read_csv(DATA_PATH / "Weather_&_Climate_Data/Processed Datas/5yeargrouped.csv")
# Check if all values are strings
# Before displaying the dataframe, convert object columns
interval_data['District'] = interval_data['District'].astype('string')  # or 'str'

# Convert 'Interval_Start' to datetime
interval_data['Interval_Start'] = pd.to_datetime(interval_data['Interval_Start'])

st.header('Daily Weather Data From The Year 1981 To 2019 Scaled to 4 years')

# Display DataFrame info() properly
with st.expander("📊 4 Years Weather Data Info"):
    buffer = StringIO()
    interval_data.info(buf=buffer)
    st.text(buffer.getvalue())

# Display a cleaner summary table
with st.expander("🧹 Clean Data Summary"):
    summary_data = {
        'Column': interval_data.columns,
        'Data Type': interval_data.dtypes,
        'Non-Null Count': interval_data.count()
    }
    st.dataframe(pd.DataFrame(summary_data))

# Option 3: Show the actual data preview
with st.expander("👀 View distributed Data"):
    st.dataframe(interval_data)

# Show basic stats
st.subheader("🌊 Basic Weather Data Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Districts", int(len(interval_data['District'].unique())))  # Explicitly cast to int
with col2:
    st.metric("Columns", len(interval_data.columns))
with col3:
    st.metric(
        label="Date Range:", 
        value=interval_data['Interval_Start'].min().strftime('%Y-%m-%d'), 
        delta=interval_data['Interval_Start'].max().strftime('%Y-%m-%d')
    )

