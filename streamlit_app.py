import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("💳 Credit Card Default Prediction System")
st.markdown("""
This application predicts whether a credit card client will default on their payment next month.
Please fill in all the client information below for an accurate prediction.
""")

# Load trained model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        # Load model
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessing objects
        with open('preprocessing.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load training data stats
        with open('training_stats.pkl', 'rb') as f:
            training_stats = pickle.load(f)
        
        return model, preprocessing, feature_names, training_stats
    except:
        st.error("Model files not found. Please ensure all model files are in the same directory.")
        st.stop()

# Load model and preprocessing
model, preprocessing, feature_names, training_stats = load_model()

# Create sidebar for input
st.sidebar.header("📋 Client Information")

# Function to create input fields
def create_input_fields():
    inputs = {}
    
    # Demographics Section
    st.sidebar.subheader("Demographic Information")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        inputs['LIMIT_BAL'] = st.number_input(
            "Credit Limit (NT$)", 
            min_value=0, 
            max_value=1000000, 
            value=150000,
            help="Total amount of credit given to the client"
        )
        
        inputs['SEX'] = st.selectbox(
            "Gender", 
            options=[1, 2], 
            format_func=lambda x: "Male" if x == 1 else "Female",
            index=1
        )
        
        inputs['EDUCATION'] = st.selectbox(
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
    
    with col2:
        inputs['MARRIAGE'] = st.selectbox(
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
        
        inputs['AGE'] = st.number_input(
            "Age (Years)", 
            min_value=21, 
            max_value=79, 
            value=35
        )
    
    # Payment History Section
    st.sidebar.subheader("Payment History (Last 6 Months)")
    st.sidebar.markdown("""
    **Payment Status Codes:**
    - -2: No consumption
    - -1: Paid in full
    - 0: Use of revolving credit
    - 1: Payment delay for 1 month
    - 2: Payment delay for 2 months
    - ... up to 8 months delay
    """)
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    pay_labels = [
        "September 2005", 
        "August 2005", 
        "July 2005", 
        "June 2005", 
        "May 2005", 
        "April 2005"
    ]
    
    for i, (col, label) in enumerate(zip(pay_cols, pay_labels)):
        cols = st.sidebar.columns(2)
        col_idx = i % 2
        with cols[col_idx]:
            inputs[col] = st.selectbox(
                f"{label}", 
                options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                index=2,  # Default to 0 (revolving credit)
                key=col
            )
    
    # Bill Amount Section
    st.sidebar.subheader("Bill Statements (NT$)")
    st.sidebar.markdown("Amount of bill statement for each month")
    
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    bill_labels = [
        "September 2005",
        "August 2005",
        "July 2005",
        "June 2005",
        "May 2005",
        "April 2005"
    ]
    
    for i, (col, label) in enumerate(zip(bill_cols, bill_labels)):
        cols = st.sidebar.columns(2)
        col_idx = i % 2
        with cols[col_idx]:
            inputs[col] = st.number_input(
                f"{label}",
                min_value=-500000,
                max_value=2000000,
                value=50000 if i < 3 else 40000,
                key=col
            )
    
    # Payment Amount Section
    st.sidebar.subheader("Previous Payments (NT$)")
    st.sidebar.markdown("Amount of previous payment for each month")
    
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    pay_amt_labels = [
        "September 2005",
        "August 2005",
        "July 2005",
        "June 2005",
        "May 2005",
        "April 2005"
    ]
    
    for i, (col, label) in enumerate(zip(pay_amt_cols, pay_amt_labels)):
        cols = st.sidebar.columns(2)
        col_idx = i % 2
        with cols[col_idx]:
            inputs[col] = st.number_input(
                f"{label}",
                min_value=0,
                max_value=2000000,
                value=5000,
                key=col
            )
    
    return inputs

# Create input fields
input_data = create_input_fields()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Summary")
    
    # Create DataFrame for display
    display_df = pd.DataFrame({
        'Feature': list(input_data.keys()),
        'Value': list(input_data.values())
    })
    
    # Categorize features for better display
    demographic_features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
    payment_history = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_amounts = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    payment_amounts = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Display in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Payment History", "Bill Amounts", "Payment Amounts"])
    
    with tab1:
        demo_df = display_df[display_df['Feature'].isin(demographic_features)].copy()
        # Format values for display
        demo_df.loc[demo_df['Feature'] == 'SEX', 'Value'] = demo_df.loc[demo_df['Feature'] == 'SEX', 'Value'].map({1: 'Male', 2: 'Female'})
        demo_df.loc[demo_df['Feature'] == 'EDUCATION', 'Value'] = demo_df.loc[demo_df['Feature'] == 'EDUCATION', 'Value'].map({
            1: 'Graduate School', 2: 'University', 3: 'High School', 
            4: 'Others', 5: 'Unknown', 6: 'Unknown', 0: 'Unknown'
        })
        demo_df.loc[demo_df['Feature'] == 'MARRIAGE', 'Value'] = demo_df.loc[demo_df['Feature'] == 'MARRIAGE', 'Value'].map({
            1: 'Married', 2: 'Single', 3: 'Others', 0: 'Unknown'
        })
        st.dataframe(demo_df.set_index('Feature'), use_container_width=True)
    
    with tab2:
        pay_df = display_df[display_df['Feature'].isin(payment_history)].copy()
        # Update labels for months
        month_labels = {
            'PAY_0': 'September 2005',
            'PAY_2': 'August 2005',
            'PAY_3': 'July 2005',
            'PAY_4': 'June 2005',
            'PAY_5': 'May 2005',
            'PAY_6': 'April 2005'
        }
        pay_df['Month'] = pay_df['Feature'].map(month_labels)
        pay_df = pay_df[['Month', 'Value']].set_index('Month')
        st.dataframe(pay_df, use_container_width=True)
    
    with tab3:
        bill_df = display_df[display_df['Feature'].isin(bill_amounts)].copy()
        bill_df['Month'] = bill_df['Feature'].map({
            'BILL_AMT1': 'September 2005',
            'BILL_AMT2': 'August 2005',
            'BILL_AMT3': 'July 2005',
            'BILL_AMT4': 'June 2005',
            'BILL_AMT5': 'May 2005',
            'BILL_AMT6': 'April 2005'
        })
        bill_df = bill_df[['Month', 'Value']]
        bill_df['Value'] = bill_df['Value'].apply(lambda x: f"NT$ {x:,}")
        st.dataframe(bill_df.set_index('Month'), use_container_width=True)
    
    with tab4:
        pay_amt_df = display_df[display_df['Feature'].isin(payment_amounts)].copy()
        pay_amt_df['Month'] = pay_amt_df['Feature'].map({
            'PAY_AMT1': 'September 2005',
            'PAY_AMT2': 'August 2005',
            'PAY_AMT3': 'July 2005',
            'PAY_AMT4': 'June 2005',
            'PAY_AMT5': 'May 2005',
            'PAY_AMT6': 'April 2005'
        })
        pay_amt_df = pay_amt_df[['Month', 'Value']]
        pay_amt_df['Value'] = pay_amt_df['Value'].apply(lambda x: f"NT$ {x:,}")
        st.dataframe(pay_amt_df.set_index('Month'), use_container_width=True)

with col2:
    st.subheader("🤖 Model Prediction")
    
    # Create a button for prediction
    if st.button("🚀 Predict Default Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing client data..."):
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure correct column order
            input_df = input_df[feature_names]
            
            # Make prediction
            try:
                # Preprocess the input
                input_processed = preprocessing.transform(input_df)
                
                # Get prediction and probabilities
                prediction = model.predict(input_processed)[0]
                probabilities = model.predict_proba(input_processed)[0]
                
                # Display results
                st.markdown("---")
                
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK: Client likely to DEFAULT**")
                    st.metric(
                        label="Default Probability", 
                        value=f"{probabilities[1]*100:.1f}%",
                        delta="High Risk"
                    )
                    
                    # Show warning signs
                    with st.expander("⚠️ Risk Factors Detected"):
                        st.markdown("""
                        Based on the input data, the following factors may contribute to the high risk:
                        - High credit utilization
                        - Late payment history
                        - Large outstanding balances
                        - Insufficient recent payments
                        """)
                else:
                    st.success("✅ **LOW RISK: Client likely to PAY**")
                    st.metric(
                        label="Default Probability", 
                        value=f"{probabilities[1]*100:.1f}%",
                        delta="Low Risk",
                        delta_color="inverse"
                    )
                    
                    # Show positive factors
                    with st.expander("✅ Positive Factors"):
                        st.markdown("""
                        The client shows several positive indicators:
                        - Good payment history
                        - Reasonable credit utilization
                        - Consistent payments
                        - Manageable debt levels
                        """)
                
                # Show probability breakdown
                st.markdown("---")
                st.subheader("Probability Breakdown")
                
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric(
                        label="Probability of PAYING",
                        value=f"{probabilities[0]*100:.1f}%",
                        delta="Safe"
                    )
                
                with prob_col2:
                    st.metric(
                        label="Probability of DEFAULTING",
                        value=f"{probabilities[1]*100:.1f}%",
                        delta="Risk"
                    )
                
                # Visualization of probabilities
                import plotly.graph_objects as go
                
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
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation based on prediction
                st.markdown("---")
                st.subheader("📋 Recommendation")
                
                if prediction == 1:
                    st.warning("""
                    **Recommended Actions:**
                    1. Increase monitoring frequency
                    2. Consider credit limit reduction
                    3. Schedule payment reminder calls
                    4. Require additional collateral
                    5. Offer payment restructuring options
                    """)
                else:
                    st.info("""
                    **Recommended Actions:**
                    1. Continue normal monitoring
                    2. Consider credit limit increase (if requested)
                    3. Maintain regular communication
                    4. Offer loyalty benefits
                    5. Regular account review
                    """)
                    
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    
    else:
        st.info("👈 Fill in all client information on the left sidebar and click the 'Predict Default Risk' button to get a prediction.")
    
    # Model information
    with st.expander("ℹ️ Model Information"):
        st.markdown(f"""
        **Model Type:** Random Forest Classifier
        **Training Accuracy:** {training_stats.get('accuracy', 'N/A')}
        **Precision:** {training_stats.get('precision', 'N/A')}
        **Recall:** {training_stats.get('recall', 'N/A')}
        **F1-Score:** {training_stats.get('f1', 'N/A')}
        
        **Dataset:** Default of Credit Card Clients
        **Samples:** 30,000 clients
        **Features:** 23 variables
        
        *Model trained on historical payment data from a large bank in Taiwan.*
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💡 <em>This prediction is based on machine learning models and should be used as a decision support tool only.</em></p>
    <p><small>For educational purposes only. Always verify with additional financial analysis.</small></p>
</div>
""", unsafe_allow_html=True)

# Add some CSS for better appearance
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        height: 3em;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
