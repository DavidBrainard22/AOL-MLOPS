import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score

def train_model():
    print("Loading data...")
    # Load data (handling potential issues with headers)
    try:
        # Detected delimiter ';' and header on 2nd line (index 1)
        df = pd.read_csv('default_of_credit_card_clients.csv', sep=';', header=1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Basic cleaning based on notebook
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    
    # Check duplicate
    df.drop_duplicates(inplace=True)

    # Define features
    numerical_columns = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    categorical_columns_ord = ['EDUCATION', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    categorical_columns_ohe = ['SEX', 'MARRIAGE']

    # New feature engineering from notebook (if any specific one needed, check notebook)
    # The notebook creates 'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT', 'PAYMENT_RATIO', 'HAS_MISSED_PAYMENT'
    # Adding these common sense features usually improves model
    df['TOTAL_BILL_AMT'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
    df['TOTAL_PAY_AMT'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)
    
    # Avoid division by zero
    df['PAYMENT_RATIO'] = df['TOTAL_PAY_AMT'] / np.where(df['TOTAL_BILL_AMT'] == 0, 1, df['TOTAL_BILL_AMT'])
    
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['HAS_MISSED_PAYMENT'] = (df[pay_status_cols] > 0).any(axis=1).astype(int)

    # Update numerical columns list with new features
    numerical_columns.extend(['TOTAL_BILL_AMT', 'TOTAL_PAY_AMT', 'PAYMENT_RATIO', 'HAS_MISSED_PAYMENT'])

    X = df.drop(columns=['default payment next month'])
    y = df['default payment next month']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Building pipeline...")
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('winsorizer', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),
                ('scaler', MinMaxScaler())
            ]), numerical_columns),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns_ord),
            ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_columns_ohe)
        ],
        remainder='passthrough'
    )

    # Model Pipeline
    # Using RandomForest as it appeared to be the chosen one in the notebook view
    pipeline_rf = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
            max_depth=10,
            n_estimators=100
        ))
    ])

    print("Training model...")
    pipeline_rf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline_rf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("Saving model...")
    joblib.dump(pipeline_rf, 'model.joblib')
    print("Model saved to model.joblib")

if __name__ == "__main__":
    train_model()
