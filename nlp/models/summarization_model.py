import os
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Ensure NLTK tokenizer is available
nltk.download('punkt')

# Summarize the article using Sumy
def summarize_article(file_path, sentence_count=5):
    with open(file_path, 'r', encoding='utf-8') as f:
        article_text = f.read().strip()

    # Use from_string with "english" tokenizer (this is correct)
    parser = PlaintextParser.from_string(article_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)

    summary_text = "\n".join(str(sentence) for sentence in summary)
    return summary_text

# Save the summary to a file
def save_summary(summary, model_name):
    # Corrected model path to ensure it saves inside 'trained_model/summarization_model'
    model_path = os.path.join('trained_model', 'summarization_model', f'{model_name}_summary.txt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved at {model_path}")


# Process all articles in the folder
def process_articles_for_summaries(raw_data_folder):
    for filename in os.listdir(raw_data_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_data_folder, filename)
            summary = summarize_article(file_path)
            print(f"Processed {filename}:\n{summary}\n")
            save_summary(summary, filename.replace('.txt', '_summary'))

if __name__ == "__main__":
    raw_data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'articles'))
    process_articles_for_summaries(raw_data_folder)
