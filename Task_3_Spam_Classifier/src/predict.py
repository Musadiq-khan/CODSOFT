
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from model import clean_text

def predict_and_evaluate():
    print("--- Starting Prediction & Evaluation ---")

    try:
        vectorizer = joblib.load('models/tfidf_spam_vectorizer.pkl')
        model = joblib.load('models/spam_classifier_nb.pkl')
        label_encoder = joblib.load('models/spam_label_encoder.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'python src/model.py' first.")
        return


    new_messages = [
        "Congratulations! You have won a £1000 gift card. Click the link now!",
        "Hey, can you call me back regarding our meeting next week?",
        "URGENT! Your bank account has been locked. Verify via link.",
        "See you at 5pm."
    ]


    df_test = pd.DataFrame({'message': new_messages})
    df_test['cleaned_message'] = df_test['message'].apply(clean_text)
    X_test_vec = vectorizer.transform(df_test['cleaned_message'])


    predictions = model.predict(X_test_vec)
    predicted_labels = label_encoder.inverse_transform(predictions)


    print("\n--- Live Prediction Results ---")
    results = pd.DataFrame({'Message': new_messages, 'Prediction': predicted_labels})
    print(results)



if __name__ == '__main__':
    predict_and_evaluate()