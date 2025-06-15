import streamlit as st
from summa.summarizer import summarize
import nltk
import logging
import os

# -------------------- Setup NLTK --------------------
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)

# -------------------- Summarization Function --------------------
def summarize_text_with_summa(text, ratio=0.3):
    try:
        if not text.strip():
            return "No input text provided."
        
        summary = summarize(text, ratio=ratio)
        return summary if summary else "Text too short or unstructured to summarize effectively."
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return f"An error occurred during summarization: {e}"

# -------------------- Load Preloaded Summaries from Files --------------------
@st.cache_data
def load_summary_outputs():
    summary_data = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    summary_dir = os.path.join(root_dir, 'nlp', 'models', 'trained_model', 'summarization_model')

    if not os.path.exists(summary_dir):
        logging.warning(f"Summarization folder '{summary_dir}' not found.")
        return summary_data

    for filename in os.listdir(summary_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(summary_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    summary_data[filename] = content
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
    return summary_data

# -------------------- Display Summary & Stats --------------------

import pandas as pd

def render_summary_text(original_text, summary_text):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📄 Original Text")
        st.text_area("Original", original_text, height=200)
    with col2:
        st.markdown("#### 📝 Summarized Text")
        st.text_area("Summary", summary_text, height=200)

    # Stats table data
    metrics = ["Word Count", "Character Count"]
    original_stats = [len(original_text.split()), len(original_text)]
    summary_stats = [len(summary_text.split()), len(summary_text)]

    df = pd.DataFrame({
        "Metric": metrics,
        "Original": original_stats,
        "Summary": summary_stats
    })
    df.insert(0, "SN", range(1, len(df) + 1))

    # Center the content using HTML and CSS
    styled_table = df.to_html(index=False, classes='centered-table')

    st.markdown("#### 📊 Summary Statistics")
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
        """,
        unsafe_allow_html=True
    )
    st.markdown(styled_table, unsafe_allow_html=True)

# -------------------- Streamlit App --------------------
def main():
    setup_nltk()
    st.title(" Text Summarization App")
    st.subheader("Choose Your Mode")

    app_mode = st.radio("Select an option:", ["Live Input ✍️", "Preloaded Files 📂"])

    if app_mode == "Live Input ✍️":
        st.markdown("#### Enter text to get a summary in real-time.")
        user_text = st.text_area("Input text:", height=200)

        if st.button("Summarize 📝"):
            if user_text.strip():
                summary_text = summarize_text_with_summa(user_text)
                render_summary_text(user_text, summary_text)
            else:
                st.warning("⚠️ Please enter text to summarize.")
    
    elif app_mode == "Preloaded Files 📂":
        st.markdown("#### Select a preprocessed file to view the summary.")
        summary_outputs = load_summary_outputs()
        if not summary_outputs:
            st.warning("⚠️ No preloaded summaries found.")
            return

        display_names = {
            file: file.replace('_', ' ').replace('.txt', '').replace('summary', '').strip()
            for file in summary_outputs
        }
        display_names = {"SELECT HERE": "SELECT HERE"} | display_names
        selected_display = st.selectbox("Select a file:", list(display_names.values()))
        selected_file = next(k for k, v in display_names.items() if v == selected_display)

        if selected_file != "SELECT HERE":
            original_text = summary_outputs[selected_file]
            summary_text = summarize_text_with_summa(original_text)
            render_summary_text(original_text, summary_text)
        else:
            st.info("ℹ️ Please select a file to view its summary.")

# -------------------- Run App --------------------
if __name__ == "__main__":
    main()