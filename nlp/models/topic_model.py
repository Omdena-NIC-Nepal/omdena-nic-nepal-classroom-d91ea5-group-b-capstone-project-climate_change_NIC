import os
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize spaCy NER model (Optional, for other purposes)
nlp = spacy.load("en_core_web_sm")

# Function to preprocess the article text (tokenization, stopword removal)
def preprocess_text(article_text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    doc = nlp(article_text)
    processed_text = ' '.join([token.text.lower() for token in doc if token.text.lower() not in stop_words and token.is_alpha])
    return processed_text

# Function to train LDA topic model
def train_topic_model(corpus, n_topics=5):
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(corpus)

    # LDA model for topic extraction
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf)
    
    return lda, tfidf_vectorizer

# Function to display the topics
def display_topics(lda_model, tfidf_vectorizer, n_words=10):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]))

# Function to save the trained topic model
def save_topic_model(lda_model, tfidf_vectorizer, model_name):
    model_path = os.path.join('trained_model', 'topic_model', f'{model_name}_topic_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the LDA model and TF-IDF vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump((lda_model, tfidf_vectorizer), f)
    print(f"Topic modeling data saved at {model_path}")

# Function to load the topic model
def load_topic_model(model_name):
    model_path = os.path.join('trained_model', 'topic_model', f'{model_name}_topic_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            lda_model, tfidf_vectorizer = pickle.load(f)
        return lda_model, tfidf_vectorizer
    else:
        print(f"Model not found at {model_path}")
        return None, None

# Example function to process all articles and extract topics
def process_articles_for_topics(raw_data_folder):
    # Read all articles and preprocess them
    corpus = []
    for filename in os.listdir(raw_data_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(raw_data_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                article_text = f.read().strip()
                preprocessed_text = preprocess_text(article_text)
                corpus.append(preprocessed_text)
    
    # Train the LDA topic model
    lda_model, tfidf_vectorizer = train_topic_model(corpus)
    
    # Display and save the topics
    display_topics(lda_model, tfidf_vectorizer)
    save_topic_model(lda_model, tfidf_vectorizer, 'climate_change')

# Example usage
if __name__ == "__main__":
    raw_data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'articles'))
    
    # Process articles and extract topics
    process_articles_for_topics(raw_data_folder)

    # Load and display the saved topic model (for verification)
    lda_model, tfidf_vectorizer = load_topic_model('climate_change')
    if lda_model:
        display_topics(lda_model, tfidf_vectorizer)
