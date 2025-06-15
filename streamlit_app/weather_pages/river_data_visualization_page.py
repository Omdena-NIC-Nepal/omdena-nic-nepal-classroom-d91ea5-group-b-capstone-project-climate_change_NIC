from pathlib import Path
import pandas as pd
import streamlit as st
from io import StringIO

# Load your data
DATA_PATH = Path(__file__).resolve().parent.parent.parent  # Go up 2 directories
rivers_data = pd.read_csv(DATA_PATH / "Weather_&_Climate_Data/Historical temperature, precipitation, and weather events/raw/glacier_and_river_data/major-rivers-in-nepal.csv")


with st.expander("📊 Rivers Data Info (Detailed)"):
    buffer = StringIO()
    rivers_data.info(buf=buffer)
    st.text(buffer.getvalue())

with st.expander("🧹 Clean Data Summary"):
    summary_data = {
        'Column': rivers_data.columns,
        'Data Type': rivers_data.dtypes,
        'Non-Null Count': rivers_data.count(),
        'Unique Values': rivers_data.nunique()
    }
    st.dataframe(pd.DataFrame(summary_data))

# Show the actual data preview
with st.expander("👀 View Rivers Data"):
    st.dataframe(rivers_data)

# Show basic stats
st.subheader("🌊 Basic River Data Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rivers", len(rivers_data))
with col2:
    st.metric("Columns", len(rivers_data.columns))
with col3:
    st.metric("Missing Values", rivers_data.isnull().sum().sum())