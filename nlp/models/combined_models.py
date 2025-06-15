# combined_model.py
import spacy
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation as LDA
from transformers import pipeline
from langdetect import detect
from summarizer import summarizer
import os
import pickle


# Load all models
def load_all_models():
    sentiment_model = TextBlob
    ner_model = spacy.load("en_core_web_sm")
    summarizer = pipeline("summarization")
    return sentiment_model, ner_model, summarizer

# Function to process each article with all models
def process_article_with_all_models(article_text):
    sentiment = TextBlob(article_text).sentiment
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article_text)
    entities = [ent.text for ent in doc.ents]
    summary = summarizer(article_text, max_length=150, min_length=50, do_sample=False)
    language = detect(article_text)
    return sentiment, entities, summary[0]['summary_text'], language

# Example usage: Process all articles in news_articles folder
if __name__ == "__main__":
    raw_data_folder = 'nlp/news_articles'
    for filename in os.listdir(raw_data_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_data_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                article_text = f.read().strip()
            sentiment, entities, summary, language = process_article_with_all_models(article_text)
            print(f"Processed {filename} with sentiment: {sentiment}, entities: {entities}, summary: {summary}, language: {language}")
