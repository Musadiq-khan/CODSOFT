
import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, 'D:/code/CODSOFT/Task_1_Movie_Genre_Prediction/data/train_data.txt')
MODEL_DIR = os.path.join(BASE_DIR, 'D:/code/CODSOFT/Task_1_Movie_Genre_Prediction/models')
os.makedirs(MODEL_DIR, exist_ok=True)

DELIMITER = '|'
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    """Clean text: lowercase, remove punctuation/numbers, and remove stopwords."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text

def train_model():
    print("Loading training data...")

    df_train = pd.read_csv(
        TRAIN_FILE,
        sep=DELIMITER,
        engine='python',
        names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'],
        skipinitialspace=True,
        encoding='latin-1', quoting = 3
    )

    df_train['CLEAN_DESCRIPTION'] = df_train['DESCRIPTION'].apply(clean_text)

  
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['GENRE'])
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    print(f"Total Genres: {len(label_encoder.classes_)}")

   
    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(df_train['CLEAN_DESCRIPTION'])

   
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(model, os.path.join(MODEL_DIR, 'genre_classifier.pkl'))

    print("✅ Model and Vectorizer saved successfully to 'models/'.")

if __name__ == '__main__':
    train_model()
