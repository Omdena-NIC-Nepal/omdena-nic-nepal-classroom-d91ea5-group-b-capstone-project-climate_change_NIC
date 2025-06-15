import os
import pickle
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Function to train the sentiment model (placeholder)
def train_sentiment_model(data):
    return {
        'textblob': TextBlob,
        'vader': vader_analyzer
    }

# Save both TextBlob and VADER sentiment results
def save_model(textblob_sentiment, vader_scores, model_name):
    model_path = os.path.join('trained_model', 'sentiment_analysis', f'{model_name}.pkl')
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

# Process article using both TextBlob and VADER
def process_article(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        article_text = f.read().strip()

    textblob_result = TextBlob(article_text).sentiment
    vader_result = vader_analyzer.polarity_scores(article_text)

    return textblob_result, vader_result

# Analyze all articles in the folder
if __name__ == "__main__":
    raw_data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'articles'))

    for filename in os.listdir(raw_data_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_data_folder, filename)
            textblob_sentiment, vader_scores = process_article(file_path)
            print(f"Processed {filename}:\n  TextBlob -> {textblob_sentiment}\n  VADER -> {vader_scores}")
            save_model(textblob_sentiment, vader_scores, filename.replace('.txt', '_sentiment_model'))

# Load and display the first saved sentiment result
def load_first_model():
    sentiment_analysis_folder = os.path.join('trained_model', 'sentiment_analysis')

    if os.path.exists(sentiment_analysis_folder):
        model_files = [f for f in os.listdir(sentiment_analysis_folder) if f.endswith('.pkl')]

        if model_files:
            model_path = os.path.join(sentiment_analysis_folder, model_files[0])
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data, model_path
        else:
            print("No model files found in the sentiment_analysis folder.")
    else:
        print(f"The folder '{sentiment_analysis_folder}' does not exist.")
    return None, None

# Display the loaded sentiment model
if __name__ == "__main__":
    model_data, model_path = load_first_model()
    if model_data:
        print(f"Loaded sentiment data from {model_path}:\n{model_data}")
    else:
        print("No model to load.")
