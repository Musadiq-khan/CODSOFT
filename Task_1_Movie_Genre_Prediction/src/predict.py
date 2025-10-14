
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import clean_text

import os
import re
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from src.model import clean_text  


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'D:/code/CODSOFT/Task_1_Movie_Genre_Prediction/data')
MODEL_DIR = os.path.join(BASE_DIR, 'D:/code/CODSOFT/Task_1_Movie_Genre_Prediction/models')

TEST_FILE = os.path.join(DATA_DIR, 'D:/code/CODSOFT/Task_1_Movie_Genre_Prediction/data/test_data.txt')
SOLUTION_FILE = os.path.join(DATA_DIR, 'D:/code/CODSOFT/Task_1_Movie_Genre_Prediction/data/test_data_solution.txt')
DELIMITER = '|'

def predict_and_evaluate():
    
    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        model = joblib.load(os.path.join(MODEL_DIR, 'genre_classifier.pkl'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    except FileNotFoundError:
        print("❌ Error: Model files not found. Please run 'python src/model.py' first.")
        return

    print("Loading test data...")
    df_test = pd.read_csv(
        TEST_FILE,
        sep=DELIMITER,
        engine='python',
        names=['ID', 'TITLE', 'DESCRIPTION'],
        skipinitialspace=True,
        encoding='latin-1', quoting = 3
    )

    df_test['CLEAN_DESCRIPTION'] = df_test['DESCRIPTION'].apply(clean_text)
    X_test = vectorizer.transform(df_test['CLEAN_DESCRIPTION'])

    predictions = model.predict(X_test)
    predicted_genres = label_encoder.inverse_transform(predictions)

    df_test['PREDICTED_GENRE'] = predicted_genres

    output_filename = os.path.join(DATA_DIR, 'predictions.txt')
    df_test[['ID', 'TITLE', 'PREDICTED_GENRE', 'DESCRIPTION']].to_csv(
        output_filename, sep=DELIMITER, index=False, header=False
    )
    print(f"✅ Predictions saved to {output_filename}")

    
    df_solution = pd.read_csv(
        SOLUTION_FILE,
        sep=DELIMITER,
        engine='python',
        names=['ID', 'TITLE', 'ACTUAL_GENRE', 'DESCRIPTION'],
        skipinitialspace=True,
        encoding='latin-1'
    )
    
    df_test['ID'] = df_test['ID'].astype(str)
    df_solution['ID'] = df_solution['ID'].astype(str)


    df_merged = pd.merge(
        df_test[['ID', 'PREDICTED_GENRE']],
        df_solution[['ID', 'ACTUAL_GENRE']],
        on='ID'
    )

    y_true = df_merged['ACTUAL_GENRE']
    y_pred = df_merged['PREDICTED_GENRE']

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == '__main__':
    predict_and_evaluate()
