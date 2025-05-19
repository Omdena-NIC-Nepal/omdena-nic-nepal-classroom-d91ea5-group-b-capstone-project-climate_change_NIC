import streamlit as st
from scripts.data_utils import load_data

def display():
    st.subheader("Data Preview")

    if "agriculture_df" not in st.session_state:
        agriculture_df = load_data("data/agricultural_data/agriculture-and-rural-development_npl.csv",skiprows=[1])
        st.session_state.agriculture_df = agriculture_df

    agriculture_df = st.session_state.agriculture_df

    st.dataframe(agriculture_df, height=200)

    st.divider()

    st.markdown("""
    ## 📊 Dataset Overview: Agricultural and Rural Development Data

    This dataset contains time-series records of agricultural and rural development indicators across countries. It includes development metrics sourced from international datasets, focusing on Nepal in this context.

    ---

    ### 1. **Country Information** 🌍  
    - **Country Name**: Name of the country (e.g., Nepal)  
    - **Country ISO3**: ISO Alpha-3 country code (e.g., NPL)

    ---

    ### 2. **Temporal Dimension** 📅  
    - **Year**: Year the data was recorded

    ---

    ### 3. **Indicator Details** 📈  
    - **Indicator Name**: Descriptive name of the indicator (e.g., Fertilizer consumption (% of fertilizer production))  
    - **Indicator Code**: Code representing the indicator (e.g., AG.CON.FERT.PT.ZS)

    ---

    ### 4. **Value** 🔢  
    - **Value**: The actual numerical value of the indicator for that year and country

    ---

    ### 5. **Column Headers Format (if raw)**  
    If using raw CSVs from development sources, columns may include standard metadata tags:  
    - `#country+name`, `#country+code`, `#date+year`, `#indicator+name`, `#indicator+code`, `#indicator+value+num`

    ---

    ### Use Cases ✅  
    This dataset supports:
    - Trend analysis over decades
    - Comparing development indicators across countries
    - Predictive modeling on agri-development metrics
    - Correlation and policy evaluation studies

    """)
    st.divider()
