import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="Prediksi Kelayakan Kartu Kredit",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main Import - Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Global Styles */
        .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #f8f9fa;
        }

        /* Header Styling */
        .main-header {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .main-header h1 {
            color: white !important;
            font-weight: 700;
        }
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* Section Cards */
        .stContainer {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border: 1px solid #e9ecef;
        }

        /* Metrics */
        .metric-card {
            background-color: #ffffff;
            border-left: 5px solid #4b6cb7;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Button Styling */
        .stButton > button {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 5px;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        /* Inputs */
        .stNumberInput, .stSelectbox {
            margin-bottom: 0.5rem;
        }
        
        /* Success/Error Message Styling */
        .prediction-success {
            padding: 1rem;
            background-color: #d4edda;
            color: #155724;
            border-radius: 5px;
            border-left: 5px solid #28a745;
            margin-top: 1rem;
        }
        .prediction-danger {
            padding: 1rem;
            background-color: #f8d7da;
            color: #721c24;
            border-radius: 5px;
            border-left: 5px solid #dc3545;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>💳 Prediksi Kelayakan Kartu Kredit</h1>
            <p>Sistem cerdas untuk analisis risiko kredit menggunakan Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

def render_feature_section(title, icon):
    st.markdown(f"### {icon} {title}")
    st.markdown("---")

def mapping_description(val, mapping):
    return f"{val} - {mapping.get(val, 'Unknown')}"
