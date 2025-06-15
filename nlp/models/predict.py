# # predict.py
# import os
# import pickle
# from textblob import TextBlob
# from langdetect import detect
# import spacy
# from transformers import pipeline

# # Load models from their respective directories
# def load_model(model_name):
#     model_path = os.path.join('models', 'sentiment_analysis', f'{model_name}.pkl')
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
#     return model

# # Function to predict using all models
# def predict(article_text):
#     sentiment_model = load_model("sentiment_model")
#     ner_model = spacy.load("en_core_web_sm")
#     summarizer = pipeline("summarization")
    
#     sentiment = sentiment_model(article_text).sentiment
#     doc = ner_model(article_text)
#     entities = [ent.text for ent in doc.ents]
#     summary = summarizer(article_text, max_length=150, min_length=50, do_sample=False)
#     language = detect(article_text)
    
#     return sentiment, entities, summary[0]['summary_text'], language

# # Example usage: Process one article
# if __name__ == "__main__":
#     article_text = "This is an example article about climate change."
#     sentiment, entities, summary, language = predict(article_text)
#     print(f"Sentiment: {sentiment}")
#     print(f"Entities: {entities}")
#     print(f"Summary: {summary}")
#     print(f"Language: {language}")



import os
import pickle
import logging
from textblob import TextBlob
import spacy
from langdetect import detect
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import CountVectorizer

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load all .pkl models
def load_all_models(base_path='trained_model'):
    models = {}
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.pkl'):
                    model_name = os.path.splitext(file)[0]
                    file_path = os.path.join(folder_path, file)
                    try:
                        with open(file_path, 'rb') as f:
                            models[model_name] = pickle.load(f)
                            logging.info(f"Loaded model: {model_name} from {file_path}")
                    except Exception as e:
                        logging.error(f"Failed to load {model_name}: {e}")
    return models

# Extract sentiment if it's a dictionary of type {'textblob': {...}, 'vader': {...}}
def extract_sentiment_scores(model_dict):
    try:
        if isinstance(model_dict, dict) and 'textblob' in model_dict and 'vader' in model_dict:
            textblob = model_dict.get('textblob', {})
            vader = model_dict.get('vader', {})
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

# Analyze the quality of the summary
def analyze_summary_quality(original_text, summary_text):
    try:
        nlp = spacy.load("en_core_web_sm")

        orig_doc = nlp(original_text)
        summ_doc = nlp(summary_text)

        orig_len = len(original_text.split())
        summ_len = len(summary_text.split())
        compression_ratio = round(orig_len / summ_len, 2) if summ_len else 0

        # Named Entity Retention
        orig_ents = {ent.text for ent in orig_doc.ents}
        summ_ents = {ent.text for ent in summ_doc.ents}
        ner_retention = round(len(orig_ents.intersection(summ_ents)) / len(orig_ents), 2) if orig_ents else 0.0

        # Keyword Coverage
        vect = CountVectorizer(stop_words='english', max_features=10)
        try:
            orig_keywords = vect.fit([original_text]).get_feature_names_out()
        except:
            orig_keywords = []
        keyword_hits = sum(1 for kw in orig_keywords if kw in summary_text.lower())

        # Readability Score
        readability = flesch_reading_ease(summary_text)

        return {
            "original_word_count": orig_len,
            "summary_word_count": summ_len,
            "compression_ratio": compression_ratio,
            "ner_retention": ner_retention,
            "keyword_coverage": keyword_hits,
            "readability_score": readability
        }

    except Exception as e:
        logging.error(f"Error analyzing summary quality: {e}")
        return {}

# Load your custom summarization model
def load_summarization_model(model_path='summarization_model/summarizer.pkl'):
    try:
        with open(model_path, 'rb') as f:
            summarizer = pickle.load(f)
            logging.info(f"Loaded custom summarization model from {model_path}")
            return summarizer
    except Exception as e:
        logging.error(f"Failed to load summarization model: {e}")
        return None

# Main prediction function
def predict(article_text):
    try:
        models = load_all_models()

        # Collect all valid sentiment scores
        sentiment_scores = {}
        for model_name, model in models.items():
            scores = extract_sentiment_scores(model)
            if scores:
                sentiment_scores[model_name] = scores

        # If no sentiment scores found, fall back to TextBlob
        if not sentiment_scores:
            polarity = TextBlob(article_text).sentiment.polarity
            subjectivity = TextBlob(article_text).sentiment.subjectivity
            sentiment_scores["textblob_fallback"] = {
                "textblob": {
                    "polarity": polarity,
                    "subjectivity": subjectivity
                },
                "vader": {
                    "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0
                }
            }

        # Load NER model
        try:
            ner_model = spacy.load("en_core_web_sm")
            doc = ner_model(article_text)
            entities = [ent.text for ent in doc.ents]
        except Exception as e:
            logging.error(f"spaCy NER model error: {e}")
            entities = []

        # Load custom summarization model and summarize
        summarizer = load_summarization_model()
        if summarizer:
            try:
                summary = summarizer(article_text)
            except Exception as e:
                logging.error(f"Summarization error with custom model: {e}")
                summary = article_text[:150] + "..."
        else:
            summary = article_text[:150] + "..."

        language = detect(article_text)

        # Analyze summary quality
        summary_metrics = analyze_summary_quality(article_text, summary)

        return sentiment_scores, entities, summary, language, summary_metrics

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None, None, None, None, None

# Example usage
if __name__ == "__main__":
    article_text = "This is an example article about climate change."
    sentiment_scores, entities, summary, language, summary_metrics = predict(article_text)

    if sentiment_scores:
        print("=== Sentiment Scores ===")
        for model, scores in sentiment_scores.items():
            print(f"\n{model}:\n  TextBlob: {scores['textblob']}\n  VADER: {scores['vader']}")

        print("\n=== Named Entities ===")
        print(entities)

        print("\n=== Summary ===")
        print(summary)

        print("\n=== Language ===")
        print(language)

        print("\n=== Summary Quality Metrics ===")
        for k, v in summary_metrics.items():
            print(f"{k}: {v}")
    else:
        print("Prediction failed.")
