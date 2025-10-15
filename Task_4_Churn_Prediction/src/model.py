
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


DATA_FILE = 'D:/code/CODSOFT/Task_4_Churn_Prediction/data/Churn_Modelling.csv'
TARGET_COLUMN = 'Exited'

def train_model():
    print("--- Starting Customer Churn Model Training ---")


    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}. Check your 'data/' folder.")
        return


    df = df.drop(['customerID'], axis=1, errors='ignore')


    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


    df.dropna(inplace=True)


    y = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else (1 if x == 1 else 0))
    X = df.drop(TARGET_COLUMN, axis=1)


    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    log_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

    print("Training Logistic Regression Model...")
    log_reg_pipeline.fit(X_train, y_train)

    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

    print("Training Random Forest Model...")
    rf_pipeline.fit(X_train, y_train)


    y_pred_log_reg = log_reg_pipeline.predict(X_test)
    auc_log_reg = roc_auc_score(y_test, log_reg_pipeline.predict_proba(X_test)[:, 1])

    y_pred_rf = rf_pipeline.predict(X_test)
    auc_rf = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1])

    print("\n--- Evaluation on Test Data ---")
    print(f"Logistic Regression AUC: {auc_log_reg:.4f}")
    print(f"Random Forest AUC: {auc_rf:.4f}")
    print("\nRandom Forest Classification Report (High Performance Model):")
    print(classification_report(y_test, y_pred_rf))


    joblib.dump(rf_pipeline, 'models/churn_predictor_rf.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')
    print("\nRandom Forest Pipeline and Feature List saved successfully to 'models/'.")

if __name__ == '__main__':
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    train_model()