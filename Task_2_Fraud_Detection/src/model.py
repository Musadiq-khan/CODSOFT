# model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

TRAIN_FILE = 'D:/code/CODSOFT/Task_2_Fraud_Detection/data/fraudTrain.csv'  # Adjust to your structure
MODEL_DIR = 'D:/code/CODSOFT/Task_2_Fraud_Detection/models'
MODEL_PATH = os.path.join(MODEL_DIR, 'fraud_detector.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_columns.pkl')

def train_model():
    print("\n--- Starting Fraud Detection Model Training ---\n")

    # Load data
    try:
        df_train = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Training file not found at {TRAIN_FILE}")
        return

    TARGET_COLUMN = 'is_fraud'
    if TARGET_COLUMN not in df_train.columns:
        print(f"❌ Error: '{TARGET_COLUMN}' not found in dataset.")
        return

    # Separate features and target
    X = df_train.drop(columns=[TARGET_COLUMN])
    y = df_train[TARGET_COLUMN]

    # Keep only numeric columns for now
    numeric_cols = X.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        print("❌ No numeric columns found in dataset.")
        return

    X = X[numeric_cols]

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train model
    print(f"Using {len(numeric_cols)} numeric features...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Validation results
    y_pred_val = model.predict(X_val)
    print("\n--- Validation Results ---")
    print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print("Classification Report:\n", classification_report(y_val, y_pred_val))

    # Save model and features
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(numeric_cols.tolist(), FEATURES_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train_model()
