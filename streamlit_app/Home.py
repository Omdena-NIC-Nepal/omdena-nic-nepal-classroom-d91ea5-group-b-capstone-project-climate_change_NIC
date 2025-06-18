import streamlit as st 
import os

if "main_section" not in st.session_state:
    st.session_state.main_section = "Select..."
if "sub_page" not in st.session_state:
    st.session_state.sub_page = "Select..."
if "page" not in st.session_state:
    st.session_state.page = "Home"

    # Sidebar Layout
st.sidebar.markdown("### Main Navigation")

main_sections = ["Climate Sections", "Weather Sections", "Socio-Economic Status"]

#Fime Mapping
PAGES = {
    "Weather Impact in Climate": "climate pages/weather_impact.py",
    "Climate Data Analysis": "climate pages/featured_data_visualization.py",
    "Climate Data Predictions": "climate pages/Predictions.py",
    "Weather Data Analysis" : "weather_pages/Data_Analysis.py",
    "Weather Data Visualization" : "weather_pages/grouped_data_visualization.py",
    "River Data Visualization": "weather_pages/river_data_visualization_page.py",
    "Socio-Economic Impact": "weather_pages/socio_economic_status.py",
    "Sentiment Analysis": "nlp_pages",
    "Language Prediction": "nlp_pages/language_prediction.py",
    "NER Prediction": "nlp_pages/ner_prediction.py",
    "Summary Details": "nlp_pages/summary_details.py"
}

subpages_mapping = {
    "Climate Sections": ["Weather Impact in Climate",
                             "Climate Data Analysis",
                            "Climate Data Predictions"

    ],
    "Weather Sections": ["Weather Data Analysis",
                        "Weather Data Visualization",
                        "River Data Visualization"
    ],
    "Socio-Economic Status": []

}
nlp_sections = [
    "Language Prediction",
    "NER Prediction",
    "Sentiment Analysis",
    "Summary Details",
]

if st.sidebar.button("🏠 Home"):
    st.session_state.main_section = "Select..."
    st.session_state.sub_page = "Select..."
    st.session_state.page = "Home"

# Select Main Section
selected_main = st.sidebar.selectbox(
    "Select Section",
    ["Select..."] + main_sections,
    index=0,
    key="main_section"
)

# Select Subpage if a Main Section is selected
if selected_main != "Select...":
    available_subpages = subpages_mapping[selected_main]
    selected_subpage = st.sidebar.selectbox(
        f"Select {selected_main} Page",
        ["Select..."] + available_subpages,
        index=0,
        key="sub_page"
    )
    
    if selected_subpage in PAGES:
        st.session_state.page = selected_subpage

# Show the District Dropdown above the NLP section if the user selects "Weather Data Visualization"
if st.session_state.page == "Weather Data Visualization || Weather Impact Assesment":
    import pandas as pd
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '../../Weather_&_Climate_Data/Weather_&_Climate_Data/Feature_Engineering/feature_engineered_climate_data_scaled.csv'))
    df = pd.read_csv(DATA_PATH)
    districts = df['district'].dropna().unique().tolist()
    selected_district = st.sidebar.selectbox("Select District", ['All'] + districts)

    if selected_district != 'All':
        df = df[df['district'] == selected_district]

# --- NLP Section Separated at Bottom ---
st.sidebar.markdown("---")
st.sidebar.markdown("### NLP Tools")

selected_nlp = st.sidebar.selectbox(
    "Select NLP Section",
    ["Select..."] + nlp_sections,
    index=0,
    key="nlp_section"
)

if selected_nlp != "Select...":
    st.session_state.page = selected_nlp

# Page Display Logic
if st.session_state.page == "Home":
    st.write("""  
    ### 🌍 Climate Prediction and Assessment App  
    Welcome to the app!  
    Navigate through the sections using the sidebar.  

    **Key Features:**
    - Vulnerability Analysis
    - Climate Trend Analysis
    - Climate Predictions
    - Socio-Economic Impact Assessment (Coming Soon!)
    - NLP Sections (Language Prediction, NER Prediction, Sentiment Analysis, Summary Details)
    """)
    st.markdown("---")
    st.warning("⚠️ Important: If the page is not redirected properly, try refreshing the browser.")
else:
    page_path = PAGES.get(st.session_state.page, None)
    if page_path:
        try:
            base_dir = os.path.dirname(__file__)
            abs_path = os.path.join(base_dir, page_path)

            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                    exec(code, globals())
            else:
                st.error(f"Error: File not found at {abs_path}")
        except Exception as e:
            st.error(f"Error loading page `{st.session_state.page}`: {str(e)}")
    else:
        st.info(f"Page `{st.session_state.page}` is a dummy page (content coming soon).")