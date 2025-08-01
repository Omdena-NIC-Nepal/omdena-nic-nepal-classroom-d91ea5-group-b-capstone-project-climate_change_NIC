import os
import logging
import streamlit as st
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load language prediction text files
@st.cache_data
def load_language_models():
    language_models = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    base_path = os.path.join(root_dir,'..', 'nlp', 'models', 'trained_model', 'language_model')

    if not os.path.exists(base_path):
        logging.warning(f"Language model folder '{base_path}' not found.")
        return language_models

    for file in os.listdir(base_path):
        if file.endswith('.txt'):
            file_path = os.path.join(base_path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    language_data = f.read()
                    language_models[file] = language_data
                    logging.info(f"Loaded language data from {file}")
            except Exception as e:
                logging.error(f"Failed to load {file}: {e}")
    return language_models

# Fallback prediction using langdetect
def predict_language_with_langdetect(text):
    try:
        return {"predicted_language": detect(text)}
    except LangDetectException as e:
        logging.error(f"Error detecting language: {e}")
        return {"predicted_language": "unknown"}

# Predict languages line-by-line from a file
@st.cache_data
def predict_languages_for_file(file_content):
    lines = [line.strip() for line in file_content.splitlines() if line.strip()]
    results = []

    for i, line in enumerate(lines, start=1):
        lang = predict_language_with_langdetect(line)['predicted_language']
        results.append((line, lang))

    return results

# Render table
def render_language_prediction_table(predictions):
    st.markdown("### 📊 Language Prediction Results")

    df = pd.DataFrame(predictions, columns=["Text", "Predicted Language"])

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

# UI for live input
def display_live_input_prediction():
    st.subheader("✍️ Predict Language from Live Input")

    article_text = st.text_area("Enter article text below", height=200)

    if st.button("Predict Language 🗣️"):
        if article_text.strip():
            result = predict_language_with_langdetect(article_text)
            render_language_prediction_table([(article_text, result['predicted_language'])])
        else:
            st.warning("Please enter some text.")

# UI for preloaded files
def display_preloaded_language_prediction():
    st.subheader("📁 Preloaded Language Prediction Fils")

    language_data = load_language_models()
    if not language_data:
        st.warning("No language prediction files found.")
        return

    files_display = {
        file: file.replace('_', ' ').replace('.txt', '').replace('language', '').strip()
        for file in language_data.keys()
    }

    files_display = {"SELECT HERE": "SELECT HERE"} | files_display
    selected_label = st.selectbox("Choose a file to view language prediction:", list(files_display.values()))

    selected_file = next(k for k, v in files_display.items() if v == selected_label)

    if selected_file != "SELECT HERE":
        file_content = language_data[selected_file]
        predictions = predict_languages_for_file(file_content)

        if predictions:
            st.markdown(f"### 🔍 Language Prediction from: `{selected_file}`")
            render_language_prediction_table(predictions)
        else:
            st.warning("The selected file is empty.")
    else:
        st.info("Please select a file.")

# Main app
def main():
    st.title("🌍 Language Prediction Tool")
    st.subheader("Choose Input Mode")

    mode = st.radio("Select an option:", ["Live Input", "Preloaded Files"])

    if mode == "Live Input":
        display_live_input_prediction()
    else:
        display_preloaded_language_prediction()

# Run
if __name__ == "__main__" or st._is_running_with_streamlit:
    main()