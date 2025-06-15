import spacy
import spacy.cli
import logging
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the spaCy model (global object)
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.error("spaCy model 'en_core_web_sm' not found. Installing it now...")
        spacy.cli.download("en_core_web_sm")  # This will download the model
        nlp = spacy.load("en_core_web_sm")  # Reload the model after downloading
        logging.info("spaCy model 'en_core_web_sm' downloaded and loaded successfully.")
    return nlp