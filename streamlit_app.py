import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from feature_engine.outliers import Winsorizer
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# Title
st.title("💳 Credit Card Default Prediction System")

# Sidebar for model training
st.sidebar.header("Model Configuration")

@st.cache_resource
def train_model():
    """Train the machine learning model"""
    try:
        # Try to load existing model
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        st.sidebar.success("✅ Pre-trained model loaded")
        return model_data
    except:
        st.sidebar.info("Training new model...")
        
        # Load data
        df = pd.read_excel('default_of_credit_card_clients.xlsx', header=1)
        df.drop(columns=['ID'], inplace=True)
        df.drop_duplicates(inplace=True)
        
        # Features and target
        X = df.drop(columns=['default payment next month'])
        y = df['default payment next month']
        
        # Define columns
        numerical_columns = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
                            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
                            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        
        # Preprocessing
        winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=numerical_columns)
        X[numerical_columns] = winsorizer.fit_transform(X[numerical_columns])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Handle imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_resampled[numerical_columns] = scaler.fit_transform(X_train_resampled[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Save model data
        model_data = {
            'model': model,
            'preprocessing': {
                'winsorizer': winsorizer,
                'scaler': scaler,
                'numerical_columns': numerical_columns
            },
            'feature_names': list(X.columns),
            'training_stats': {
                'accuracy': f"{accuracy:.4f}",
                'precision': f"{precision:.4f}",
                'recall': f"{recall:.4f}",
                'f1': f"{f1:.4f}"
            }
        }
        
        # Save to file
        with open('model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        st.sidebar.success("✅ Model trained and saved")
        return model_data

# Load or train model
model_data = train_model()
model = model_data['model']
preprocessing = model_data['preprocessing']
feature_names = model_data['feature_names']
training_stats = model_data['training_stats']

# Main app
st.markdown("""
### Predict whether a credit card client will default on their payment next month
Please fill in all the client information below for an accurate prediction.
""")

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic Information")
    
    LIMIT_BAL = st.number_input(
        "Credit Limit (NT$)", 
        min_value=0, 
        max_value=1000000, 
        value=150000,
        help="Total amount of credit given to the client"
    )
    
    SEX = st.selectbox(
        "Gender", 
        options=[1, 2], 
        format_func=lambda x: "Male" if x == 1 else "Female",
        index=1
    )
    
    EDUCATION = st.selectbox(
        "Education Level", 
        options=[1, 2, 3, 4, 5, 6, 0], 
        format_func=lambda x: {
            1: "Graduate School",
            2: "University",
            3: "High School",
            4: "Others",
            5: "Unknown",
            6: "Unknown",
            0: "Unknown"
        }[x],
        index=1
    )
    
    MARRIAGE = st.selectbox(
        "Marital Status", 
        options=[1, 2, 3, 0], 
        format_func=lambda x: {
            1: "Married",
            2: "Single",
            3: "Others",
            0: "Unknown"
        }[x],
        index=1
    )
    
    AGE = st.number_input(
        "Age (Years)", 
        min_value=21, 
        max_value=79, 
        value=35
    )

with col2:
    st.subheader("Recent Payment Status")
    st.markdown("""
    **Status Codes:**
    - -2: No consumption
    - -1: Paid in full
    - 0: Use of revolving credit
    - 1-8: Months of payment delay
    """)
    
    PAY_0 = st.selectbox("September 2005", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
    PAY_2 = st.selectbox("August 2005", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
    PAY_3 = st.selectbox("July 2005", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
    PAY_4 = st.selectbox("June 2005", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
    PAY_5 = st.selectbox("May 2005", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
    PAY_6 = st.selectbox("April 2005", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)

# Bill amounts
st.subheader("Bill Statement Amounts (NT$)")
bill_cols = st.columns(6)
bill_values = []
bill_labels = ['Sept 2005', 'Aug 2005', 'July 2005', 'June 2005', 'May 2005', 'Apr 2005']
bill_defaults = [50000, 48000, 46000, 43000, 40000, 38000]

for i, col in enumerate(bill_cols):
    with col:
        value = st.number_input(
            bill_labels[i],
            min_value=-500000,
            max_value=2000000,
            value=bill_defaults[i],
            key=f"bill_{i}"
        )
        bill_values.append(value)

BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6 = bill_values

# Payment amounts
st.subheader("Previous Payment Amounts (NT$)")
pay_amt_cols = st.columns(6)
pay_amt_values = []
pay_amt_labels = ['Sept 2005', 'Aug 2005', 'July 2005', 'June 2005', 'May 2005', 'Apr 2005']

for i, col in enumerate(pay_amt_cols):
    with col:
        value = st.number_input(
            pay_amt_labels[i],
            min_value=0,
            max_value=2000000,
            value=5000,
            key=f"pay_amt_{i}"
        )
        pay_amt_values.append(value)

PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6 = pay_amt_values

# Prediction button
if st.button("🚀 Predict Default Risk", type="primary", use_container_width=True):
    # Prepare input data
    input_dict = {
        'LIMIT_BAL': LIMIT_BAL,
        'SEX': SEX,
        'EDUCATION': EDUCATION,
        'MARRIAGE': MARRIAGE,
        'AGE': AGE,
        'PAY_0': PAY_0,
        'PAY_2': PAY_2,
        'PAY_3': PAY_3,
        'PAY_4': PAY_4,
        'PAY_5': PAY_5,
        'PAY_6': PAY_6,
        'BILL_AMT1': BILL_AMT1,
        'BILL_AMT2': BILL_AMT2,
        'BILL_AMT3': BILL_AMT3,
        'BILL_AMT4': BILL_AMT4,
        'BILL_AMT5': BILL_AMT5,
        'BILL_AMT6': BILL_AMT6,
        'PAY_AMT1': PAY_AMT1,
        'PAY_AMT2': PAY_AMT2,
        'PAY_AMT3': PAY_AMT3,
        'PAY_AMT4': PAY_AMT4,
        'PAY_AMT5': PAY_AMT5,
        'PAY_AMT6': PAY_AMT6
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]
    
    # Preprocess
    numerical_columns = preprocessing['numerical_columns']
    winsorizer = preprocessing['winsorizer']
    scaler = preprocessing['scaler']
    
    # Handle outliers
    input_df[numerical_columns] = winsorizer.transform(input_df[numerical_columns])
    
    # Scale numerical features
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    # Display results
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ **HIGH RISK: Client likely to DEFAULT**")
        else:
            st.success("✅ **LOW RISK: Client likely to PAY**")
        
        st.metric(
            label="Default Probability", 
            value=f"{probabilities[1]*100:.1f}%",
            delta="High Risk" if prediction == 1 else "Low Risk",
            delta_color="inverse" if prediction == 0 else "normal"
        )
    
    with col2:
        # Probability visualization
        fig = go.Figure(data=[
            go.Bar(
                x=['Pay', 'Default'],
                y=[probabilities[0], probabilities[1]],
                marker_color=['green', 'red'],
                text=[f'{probabilities[0]*100:.1f}%', f'{probabilities[1]*100:.1f}%'],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("📋 Recommendations")
    if prediction == 1:
        st.warning("""
        **Consider these actions:**
        1. Increase monitoring frequency
        2. Consider credit limit reduction
        3. Schedule payment reminder calls
        4. Require additional collateral
        5. Offer payment restructuring options
        """)
    else:
        st.info("""
        **Recommended actions:**
        1. Continue normal monitoring
        2. Consider credit limit increase (if requested)
        3. Maintain regular communication
        4. Offer loyalty benefits
        5. Regular account review
        """)

# Model info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.markdown(f"""
**Model Type:** Random Forest Classifier
**Accuracy:** {training_stats['accuracy']}
**Precision:** {training_stats['precision']}
**Recall:** {training_stats['recall']}
**F1-Score:** {training_stats['f1']}

**Features:** 23 variables
**Training Data:** 30,000 clients
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💡 <em>This prediction is based on machine learning models and should be used as a decision support tool only.</em></p>
    <p><small>For educational purposes only. Always verify with additional financial analysis.</small></p>
</div>
""", unsafe_allow_html=True)

# Add CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        height: 3em;
        font-size: 1.2em;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)
