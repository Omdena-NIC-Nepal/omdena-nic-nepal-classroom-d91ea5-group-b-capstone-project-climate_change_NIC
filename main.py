import streamlit as st

# Page Configuration
PAGE_CONFIG = {
    "page_title": "Nepal Climate-Agriculture Dashboard",
    "layout": "wide"
}

# Navigation Options
NAVIGATION_OPTIONS = [
    "About",
    "Data Overview",
    "Climate Analysis",
    "Feature Engineering",
    "Model Development",
    "Predictions",
    "NLP Insights"
]

# Initialize page configuration
st.set_page_config(**PAGE_CONFIG)

# Main Title
st.header("🌍 Nepal Climate-Agriculture Analysis Dashboard")

# Sidebar Navigation
selected_page = st.sidebar.radio(
    "Navigation",
    NAVIGATION_OPTIONS,
    index=1
)

# Page Routing
def load_page(page_name):
    """Load and display the selected page content"""
    page_mapping = {
        "About": "about_page",
        "Data Overview": "overview_page",
        "Climate Analysis": "eda_page",
        "Feature Engineering": "feat_engineering_page",
        "Model Development": "modeltrain_page",
        "Predictions": "prediction_page",
        "NLP Insights": "NLP"
    }
    
    if page_name in page_mapping:
        module_name = page_mapping[page_name]
        module = __import__(f"tabs.{module_name}", fromlist=["display"])
        module.display()

# Display selected page
load_page(selected_page)

# Footer
if selected_page != "About":
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Version 1.0 | Developed by <b>Sadikshya Subedi</b></p>
            <p>
                <a href="https://www.linkedin.com/in/sadikshya-subedi-70b0842a5/" target="_blank" style="text-decoration: none;">LinkedIn</a> |
                <a href="https://github.com/Mathcurio01" target="_blank" style="text-decoration: none;">GitHub</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
