import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


DATA_FILE = r'D:/code/CODSOFT/Task_4_Churn_Prediction/data/Churn_Modelling.csv'
TARGET_COLUMN = 'Exited'

def train_model():
    print("--- Starting Customer Churn Model Training ---")

    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}.")
        return




    ID_COLS = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(ID_COLS, axis=1, errors='ignore')


    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


    df.dropna(inplace=True)


    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)


    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

    print("Training Random Forest Pipeline...")
    rf_pipeline.fit(X_train, y_train)




    joblib.dump(rf_pipeline, 'models/churn_predictor_rf.pkl')

    print("\nRandom Forest Pipeline saved successfully to 'models/'.")

if __name__ == '__main__':
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    train_model()