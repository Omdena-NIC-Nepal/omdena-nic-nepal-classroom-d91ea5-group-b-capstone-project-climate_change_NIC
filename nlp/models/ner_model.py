import os
import pickle
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Initialize VADER analyzer and spaCy NER model
vader_analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")  # spaCy model for Named Entity Recognition (NER)

# Function to train the sentiment model (placeholder)
def train_sentiment_model(data):
    return {
        'textblob': TextBlob,
        'vader': vader_analyzer
    }

# Save both TextBlob and VADER sentiment results
def save_sentiment_model(textblob_sentiment, vader_scores, model_name):
    model_path = os.path.join('trained_model', 'sentiment_analysis', f'{model_name}_sentiment.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    sentiment_data = {
        'textblob': {
            'polarity': textblob_sentiment.polarity,
            'subjectivity': textblob_sentiment.subjectivity
        },
        'vader': vader_scores
    }

    with open(model_path, 'wb') as f:
        pickle.dump(sentiment_data, f)
    print(f"Sentiment analysis data saved at {model_path}")

# Save NER entities
def save_ner_entities(ner_entities, model_name):
    model_path = os.path.join('trained_model', 'ner', f'{model_name}_ner.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(ner_entities, f)
    print(f"NER data saved at {model_path}")

# Function to extract NER entities from text (locations, organizations, persons)
def extract_ner_entities(article_text):
    doc = nlp(article_text)
    entities = {
        'locations': [ent.text for ent in doc.ents if ent.label_ == 'GPE'],  # Geopolitical Entities (locations)
        'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],  # Organizations
        'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],  # Persons
    }
    return entities

# Process article using both TextBlob, VADER, and NER (separate processes)
def process_sentiment_and_ner(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        article_text = f.read().strip()

    # Sentiment analysis with TextBlob and VADER
    textblob_result = TextBlob(article_text).sentiment
    vader_result = vader_analyzer.polarity_scores(article_text)

    # NER processing (separate from sentiment)
    ner_entities = extract_ner_entities(article_text)

    return textblob_result, vader_result, ner_entities

# Analyze all articles in the folder
if __name__ == "__main__":
    raw_data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'articles'))

    for filename in os.listdir(raw_data_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_data_folder, filename)
            textblob_sentiment, vader_scores, ner_entities = process_sentiment_and_ner(file_path)

            # Print results for verification
            print(f"Processed {filename}:\n  TextBlob -> {textblob_sentiment}\n  VADER -> {vader_scores}\n  NER -> {ner_entities}")

            # Save both sentiment and NER results separately
            save_sentiment_model(textblob_sentiment, vader_scores, filename.replace('.txt', '_sentiment'))
            save_ner_entities(ner_entities, filename.replace('.txt', '_ner'))

# Load and display the first saved sentiment model
def load_first_model(model_type='sentiment'):
    folder = os.path.join('trained_model', model_type)
    
    if os.path.exists(folder):
        model_files = [f for f in os.listdir(folder) if f.endswith('.pkl')]

        if model_files:
            model_path = os.path.join(folder, model_files[0])
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data, model_path
        else:
            print(f"No {model_type} model files found in the folder.")
    else:
        print(f"The folder '{folder}' does not exist.")
    return None, None

# Display the loaded sentiment or NER model
if __name__ == "__main__":
    # Example: Load and display the first sentiment model
    model_data, model_path = load_first_model('sentiment')
    if model_data:
        print(f"Loaded sentiment data from {model_path}:\n{model_data}")
    else:
        print("No sentiment model to load.")

    # Example: Load and display the first NER model
    ner_data, ner_path = load_first_model('ner')
    if ner_data:
        print(f"Loaded NER data from {ner_path}:\n{ner_data}")
    else:
        print("No NER model to load.")
