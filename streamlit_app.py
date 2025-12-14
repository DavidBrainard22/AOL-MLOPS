# =========================================================
# Credit Card Default Prediction - Streamlit App
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Prediksi Default Kartu Kredit",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.main-header {
    font-size: 2.4rem;
    font-weight: bold;
    color: #1E3A8A;
    text-align: center;
}
.sub-header {
    font-size: 1.4rem;
    color: #2563EB;
    margin-top: 1rem;
}
.card {
    padding: 1.5rem;
    border-radius: 10px;
    background-color: #F8FAFC;
    box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("default_of_credit_card_clients.csv", header=1)

        # 🔑 FIX UTAMA
        df.columns = df.columns.str.strip()

        df = df.rename(columns={
            "default payment next month": "default_payment"
        })

        df = df.drop(columns=["ID"], errors="ignore")

        return df

    except Exception:
        st.warning("Dataset asli tidak ditemukan. Menggunakan data contoh.")

        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            "LIMIT_BAL": np.random.randint(10000, 1000000, n),
            "SEX": np.random.choice([1, 2], n),
            "EDUCATION": np.random.choice([1, 2, 3, 4], n),
            "MARRIAGE": np.random.choice([1, 2, 3], n),
            "AGE": np.random.randint(21, 70, n),
            "PAY_0": np.random.choice([-1, 0, 1, 2], n),
            "PAY_2": np.random.choice([-1, 0, 1, 2], n),
            "PAY_3": np.random.choice([-1, 0, 1, 2], n),
            "PAY_4": np.random.choice([-1, 0, 1, 2], n),
            "PAY_5": np.random.choice([-1, 0, 1, 2], n),
            "PAY_6": np.random.choice([-1, 0, 1, 2], n),
            "BILL_AMT1": np.random.randint(0, 200000, n),
            "BILL_AMT2": np.random.randint(0, 200000, n),
            "BILL_AMT3": np.random.randint(0, 200000, n),
            "BILL_AMT4": np.random.randint(0, 200000, n),
            "BILL_AMT5": np.random.randint(0, 200000, n),
            "BILL_AMT6": np.random.randint(0, 200000, n),
            "PAY_AMT1": np.random.randint(0, 50000, n),
            "PAY_AMT2": np.random.randint(0, 50000, n),
            "PAY_AMT3": np.random.randint(0, 50000, n),
            "PAY_AMT4": np.random.randint(0, 50000, n),
            "PAY_AMT5": np.random.randint(0, 50000, n),
            "PAY_AMT6": np.random.randint(0, 50000, n),
            "default_payment": np.random.choice([0, 1], n, p=[0.78, 0.22])
        })

        return df

# =========================================================
# TRAIN MODEL
# =========================================================
@st.cache_resource
def train_model(df):
    X = df.drop("default_payment", axis=1)
    y = df["default_payment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns

# =========================================================
# LOAD
# =========================================================
df = load_data()
model, scaler, feature_columns = train_model(df)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📊 Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["🏠 Dashboard", "📈 EDA", "🤖 Prediksi", "📋 Dataset"]
)

# =========================================================
# DASHBOARD
# =========================================================
if menu == "🏠 Dashboard":
    st.markdown("<div class='main-header'>💳 Credit Card Default Prediction</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Nasabah", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['default_payment'].mean()*100:.2f}%")
    col3.metric("Rata-rata Limit", f"{df['LIMIT_BAL'].mean():,.0f}")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Ringkasan Proyek")
    st.markdown("""
    Aplikasi ini digunakan untuk memprediksi kemungkinan **default pembayaran kartu kredit**
    menggunakan model **Random Forest** berdasarkan data historis nasabah.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# EDA
# =========================================================
elif menu == "📈 EDA":
    st.markdown("<div class='main-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    numeric_cols = [c for c in ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"] if c in df.columns]
    feature = st.selectbox("Pilih fitur numerik:", numeric_cols)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[feature], bins=30, edgecolor="black")
    ax.set_title(f"Distribusi {feature}")
    st.pyplot(fig)

    st.markdown(f"""
    **Skewness:** {df[feature].skew():.2f}  
    **Kurtosis:** {df[feature].kurtosis():.2f}
    """)

# =========================================================
# PREDICTION
# =========================================================
elif menu == "🤖 Prediksi":
    st.markdown("<div class='main-header'>🤖 Prediksi Default</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        LIMIT_BAL = st.number_input("Limit Kredit", 0, 1000000, 200000)
        AGE = st.slider("Usia", 21, 70, 35)
        SEX = st.selectbox("Jenis Kelamin", [1, 2], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")

    with col2:
        BILL_AMT1 = st.number_input("Tagihan Bulan Terakhir", 0, 500000, 50000)
        PAY_AMT1 = st.number_input("Pembayaran Bulan Terakhir", 0, 500000, 5000)
        PAY_0 = st.selectbox("Status Pembayaran Terakhir", [-1, 0, 1, 2])

    if st.button("Prediksi"):
        input_df = pd.DataFrame([{
            "LIMIT_BAL": LIMIT_BAL,
            "SEX": SEX,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": AGE,
            "PAY_0": PAY_0,
            "PAY_2": 0,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": BILL_AMT1,
            "BILL_AMT2": BILL_AMT1,
            "BILL_AMT3": BILL_AMT1,
            "BILL_AMT4": BILL_AMT1,
            "BILL_AMT5": BILL_AMT1,
            "BILL_AMT6": BILL_AMT1,
            "PAY_AMT1": PAY_AMT1,
            "PAY_AMT2": PAY_AMT1,
            "PAY_AMT3": PAY_AMT1,
            "PAY_AMT4": PAY_AMT1,
            "PAY_AMT5": PAY_AMT1,
            "PAY_AMT6": PAY_AMT1
        }])[feature_columns]

        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Berisiko DEFAULT (Probabilitas: {prob:.2%})")
        else:
            st.success(f"✅ Tidak Default (Probabilitas: {1-prob:.2%})")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# DATASET INFO
# =========================================================
else:
    st.markdown("<div class='main-header'>📋 Informasi Dataset</div>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2024 Credit Card Default Prediction App</p>",
    unsafe_allow_html=True
)
