import os
import pickle
import logging
import streamlit as st
from textblob import TextBlob
import pandas as pd


# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load sentiment outputs from .pkl files
@st.cache_data
def load_sentiment_outputs():
    sentiment_models = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    base_path = os.path.join(root_dir, '..','nlp', 'models', 'trained_model', 'sentiment_analysis')

    if not os.path.exists(base_path):
        logging.warning(f"Sentiment model folder '{base_path}' not found.")
        return sentiment_models

    for file in os.listdir(base_path):
        if file.endswith('.pkl'):
            file_path = os.path.join(base_path, file)
            try:
                with open(file_path, 'rb') as f:
                    sentiment_data = pickle.load(f)
                    sentiment_models[file] = sentiment_data
                    logging.info(f"Loaded sentiment data from {file}")
            except Exception as e:
                logging.error(f"Failed to load {file}: {e}")
    return sentiment_models

# Extract sentiment scores from model dictionary
def extract_sentiment_scores(model_dict):
    try:
        if isinstance(model_dict, dict) and 'textblob' in model_dict and 'vader' in model_dict:
            textblob = model_dict.get("textblob", {})
            vader = model_dict.get("vader", {})
            return {
                "textblob": {
                    "polarity": textblob.get("polarity", 0.0),
                    "subjectivity": textblob.get("subjectivity", 0.0)
                },
                "vader": {
                    "neg": vader.get("neg", 0.0),
                    "neu": vader.get("neu", 1.0),
                    "pos": vader.get("pos", 0.0),
                    "compound": vader.get("compound", 0.0)
                }
            }
    except Exception as e:
        logging.error(f"Error extracting sentiment: {e}")
    return None

# Fallback sentiment analysis using TextBlob
def analyze_with_textblob(text):
    blob = TextBlob(text).sentiment
    return {
        "textblob": {
            "polarity": blob.polarity,
            "subjectivity": blob.subjectivity
        },
        "vader": {
            "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0
        }
    }


# Render sentiment results as table with centered content
def render_sentiment_table(scores: dict):
    st.markdown("### 📊 Sentiment Scores")

    # Prepare the data
    data = {
        "Metric": [
            "TextBlob Polarity", "TextBlob Subjectivity",
            "VADER Negative", "VADER Neutral", "VADER Positive", "VADER Compound"
        ],
        "Score": [
            round(scores['textblob']['polarity'], 2),
            round(scores['textblob']['subjectivity'], 2),
            round(scores['vader']['neg'], 2),
            round(scores['vader']['neu'], 2),
            round(scores['vader']['pos'], 2),
            round(scores['vader']['compound'], 2)
        ]
    }

    df = pd.DataFrame(data)
    df.insert(0, "SN", range(1, len(df) + 1))

    # Center the content using HTML and CSS
    styled_table = df.to_html(index=False, classes='centered-table')

    st.markdown(
        """
        <style>
        .centered-table {
            width: 100%;
            border-collapse: collapse;
        }
        .centered-table th, .centered-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .centered-table th {
            background-color: #f2f2f2;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown(styled_table, unsafe_allow_html=True)


# Live input UI
def display_live_input_analysis():
    st.subheader("✍️ Analyze Live Input")

    article_text = st.text_area("Enter article text below", height=200)

    if st.button("Analyze Sentiment"):
        if article_text.strip():
            result = analyze_with_textblob(article_text)
            render_sentiment_table(result)
        else:
            st.warning("Please enter some text.")

# Preloaded file UI
def display_preloaded_sentiment():
    st.subheader("📁 Preloaded Sentiment Files")

    sentiment_data = load_sentiment_outputs()
    if not sentiment_data:
        st.warning("No sentiment files found.")
        return

    # Format file names
    files_display = {
        file: file.replace('_', ' ').replace('.pkl', '').replace('sentiment', '').strip()
        for file in sentiment_data.keys()
    }

    files_display = {"SELECT HERE": "SELECT HERE"} | files_display
    selected_label = st.selectbox("Choose a file to view sentiment results:", list(files_display.values()))

    selected_file = next(k for k, v in files_display.items() if v == selected_label)

    if selected_file != "SELECT HERE":
        scores = extract_sentiment_scores(sentiment_data[selected_file])
        if scores:
            st.markdown(f"### 🔍 Sentiment from: `{selected_file}`")
            render_sentiment_table(scores)
        else:
            st.error("Invalid data in the selected file.")
    else:
        st.info("Please select a file.")

# Main app
def main():
    st.title(" Sentiment Analysis Tool")
    st.subheader("Choose Input Mode")

    mode = st.radio("Select an option:", ["Live Input", "Preloaded Files"])

    if mode == "Live Input":
        display_live_input_analysis()
    else:
        display_preloaded_sentiment()

# Run
if __name__ == "__main__" or st._is_running_with_streamlit:
    main()