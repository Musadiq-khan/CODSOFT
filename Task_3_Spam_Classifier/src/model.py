import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import nltk


try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


DATA_FILE = 'D:/code/CODSOFT/Task_3_Spam_Classifier/data/spam.csv'
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """Simple text cleaning for SMS: lowercasing, removing punctuation, and stop word removal."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text

def train_model():
    print("--- Starting Spam Classification Model Training ---")


    try:

        df = pd.read_csv(DATA_FILE, encoding='latin-1')

        df = df.iloc[:, [0, 1]].copy()
        df.columns = ['label', 'message']
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}. Check your 'data/' folder.")
        return


    print("Cleaning SMS messages...")
    df['cleaned_message'] = df['message'].apply(clean_text)


    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    y = df['label_encoded']


    X_train_msg, X_test_msg, y_train, y_test = train_test_split(
        df['cleaned_message'], y, test_size=0.2, stratify=y, random_state=42
    )


    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train_msg)


    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)


    joblib.dump(vectorizer, 'models/tfidf_spam_vectorizer.pkl')
    joblib.dump(model, 'models/spam_classifier_nb.pkl')
    joblib.dump(label_encoder, 'models/spam_label_encoder.pkl')
    print("Model and Vectorizer saved successfully to 'models/'.")

if __name__ == '__main__':
    import os
    if not os.path.exists('models'):
        os.makedirs('models')

    train_model()