import pickle
import streamlit as st 
import spacy 
import logging 
import os
import pandas as pd

# Supported entity types
SUPPORTED_ENTITY_LABELS = [
    "PERSON", "ORG", "GPE", "LOC", "NORP", "MONEY", "DATE", "TIME", "PERCENT", "FAC"
]
nlp = spacy.load("en_core_web_sm")
@st.cache_data
def load_ner_outputs():
    ner_models = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    base_path = os.path.join(root_dir,'..', 'nlp', 'models', 'trained_model', 'language_model')

    if not os.path.exists(base_path):
        logging.warning(f"NER output folder '{base_path}' not found.")
        return ner_models
    
    for filename in os.listdir(base_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(base_path, filename)
            try:
                with open(file_path, 'rb') as f:
                    ner_data = pickle.load(f)
                    logging.info(f"Loaded NER data from {filename}")
                    ner_models[filename] = ner_data
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
    return ner_models

# Perform NER using spaCy
def perform_ner_on_input(text):
    doc = nlp(text)
    entities = {label: [] for label in SUPPORTED_ENTITY_LABELS}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return {k: v for k, v in entities.items() if v}

def render_entity_table(entities: dict):
    if not entities:
        st.info("No named entities found.")
        return
    
    data = [{"Entity Type": label, "Entity": item}
            for label, items in entities.items() for item in set(items)]
    df = pd.DataFrame(data)
    df.insert(0, "SN", range(1, len(df) + 1))

      # Convert to styled HTML table
    styled_table = df.to_html(index=False, classes="styled-table", escape=False)

# CSS styling
    table_css = """
    <style>
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 16px;
            font-family: sans-serif;
        }
        .styled-table thead tr {
            background-color: #ddd;
            text-align: center;
        }
        .styled-table th, .styled-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .styled-table tbody tr:nth-child(even) {
            background-color: #a19e8a;
        }
        .styled-table tbody tr:hover {
            background-color: #7ccbeb;
        }
    </style>
    """

    # Display in Streamlit with styling
    st.markdown(table_css, unsafe_allow_html=True)
    st.markdown(styled_table, unsafe_allow_html=True)

def display_ner_for_user_input():
    st.title("Named Entity Recognition (NER) - Live Input")

    user_text = st.text_area("Enter text for NER analysis", height=200)

    if st.button("Analyze NER"):
        if user_text.strip():
            entities = perform_ner_on_input(user_text)
            st.subheader(" Extracted Entities")
            render_entity_table(entities)
        else:
            st.warning("Please enter some text to analyze.")

def display_ner_from_saved_models():
    st.title("📁 NER from Preprocessed Files")

    ner_outputs = load_ner_outputs()
    if not ner_outputs:
        st.warning("No preloaded NER outputs found.")
        return        
    
    files_with_spaces = {
        file: file.replace('_', ' ').replace('.pkl', '').replace('ner', '').strip() 
        for file in ner_outputs.keys()
    }

    files_with_spaces = {"SELECT HERE": "SELECT HERE"} | files_with_spaces

    selected_file_display_name = st.selectbox("Select a preprocessed NER file:", list(files_with_spaces.values()))

    selected_file = next(key for key, value in files_with_spaces.items() if value == selected_file_display_name)

    if selected_file != "SELECT HERE":
        entities = ner_outputs[selected_file]
        if isinstance(entities, dict):
            st.subheader(f"🔍 Entities in: {selected_file}")
            render_entity_table(entities)
        else:
            st.error("Invalid structure in the selected file.")
    else:
        # If the placeholder is selected, do nothing or display a message
        st.info("Please select a file to view extracted NER entities.")

# Enhanced UI with sections
def main():
    st.title(" Named Entity Recognition (NER) App")
    st.subheader(" Choose Your Mode")

    app_mode = st.radio("Select an option:", ["Live Input", "Preloaded Files"])

    if app_mode == "Live Input":
        st.markdown("#### Enter text to analyze entities in real-time.")
        display_ner_for_user_input()
    else:
        st.markdown("#### Select a preprocessed file to view extracted NER entities.")
        display_ner_from_saved_models()

if __name__ == "__main__" or st._is_running_with_streamlit:
    main()