# predict.py
import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Use consistent local paths
TEST_FILE = 'D:/code/CODSOFT/Task_2_Fraud_Detection/data/fraudTrain.csv'  # Use same dataset if no separate test file
MODEL_DIR = 'D:/code/CODSOFT/Task_2_Fraud_Detection/models'
MODEL_PATH = os.path.join(MODEL_DIR, 'fraud_detector.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_columns.pkl')

def predict_and_evaluate():
    print("\n--- Starting Prediction & Evaluation ---\n")

    # Load model and feature columns
    try:
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
    except FileNotFoundError:
        print("❌ Error: Model files not found. Please run 'model.py' first.")
        return

    # Load test data
    try:
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Test file not found at {TEST_FILE}.")
        return

    TARGET_COLUMN = 'is_fraud'
    if TARGET_COLUMN not in df_test.columns:
        print(f"❌ Error: Column '{TARGET_COLUMN}' not found in dataset.")
        return

    # Prepare test data
    X_test = df_test.drop(columns=[TARGET_COLUMN])
    y_true = df_test[TARGET_COLUMN]
    X_test = X_test.fillna(X_test.median(numeric_only=True))

    # Ensure correct column alignment
    X_test = X_test[feature_cols]

    # Predict
    print("Generating predictions...")
    y_pred = model.predict(X_test)

    # Evaluate
    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

if __name__ == '__main__':
    predict_and_evaluate()
