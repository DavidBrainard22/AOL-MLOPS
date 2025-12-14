# =========================================================
# STREAMLIT APP - CREDIT CARD DEFAULT PREDICTION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Prediksi Default Kartu Kredit",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# LOAD DATA (ANTI ERROR VERSION)
# =========================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("default_of_credit_card_clients.csv", header=1)

        # 🔑 FIX UTAMA (WAJIB)
        df.columns = df.columns.str.strip()

        # Rename target
        if "default payment next month" in df.columns:
            df = df.rename(columns={
                "default payment next month": "default_payment"
            })

        # Drop ID jika ada
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        return df

    except Exception as e:
        st.warning("Dataset tidak ditemukan, menggunakan data dummy.")

        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            "LIMIT_BAL": np.random.randint(10000, 500000, n),
            "SEX": np.random.choice([1, 2], n),
            "EDUCATION": np.random.choice([1, 2, 3, 4], n),
            "MARRIAGE": np.random.choice([1, 2, 3], n),
            "AGE": np.random.randint(21, 70, n),
            "PAY_0": np.random.choice([-1, 0, 1, 2], n),
            "BILL_AMT1": np.random.randint(0, 200000, n),
            "PAY_AMT1": np.random.randint(0, 50000, n),
            "default_payment": np.random.choice([0, 1], n, p=[0.78, 0.22])
        })

        return df


# =========================================================
# TRAIN MODEL (ANTI default_payment ERROR)
# =========================================================
@st.cache_resource
def train_model(df):
    target_col = "default_payment"

    if target_col not in df.columns:
        st.error("Kolom target 'default_payment' tidak ditemukan.")
        st.stop()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns.tolist()


# =========================================================
# LOAD EVERYTHING
# =========================================================
df = load_data()
model, scaler, feature_columns = train_model(df)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📊 Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["🏠 Dashboard", "📈 EDA", "🤖 Prediksi"]
)

# =========================================================
# DASHBOARD
# =========================================================
if menu == "🏠 Dashboard":
    st.title("💳 Prediksi Default Kartu Kredit")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Data", f"{len(df):,}")

    with col2:
        default_rate = df["default_payment"].mean() * 100
        st.metric("Default Rate", f"{default_rate:.2f}%")

    with col3:
        st.metric("Rata-rata Limit", f"{df['LIMIT_BAL'].mean():,.0f}")

    st.markdown("### Preview Dataset")
    st.dataframe(df.head(), use_container_width=True)


# =========================================================
# EDA
# =========================================================
elif menu == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    numeric_cols = [c for c in df.columns if c != "default_payment"]

    selected = st.selectbox("Pilih fitur numerik", numeric_cols)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[selected], bins=30, edgecolor="black")
    ax.set_title(f"Distribusi {selected}")
    st.pyplot(fig)

    st.markdown("### Korelasi dengan Target")
    corr = df.corr(numeric_only=True)["default_payment"].sort_values()
    st.dataframe(corr)


# =========================================================
# PREDICTION
# =========================================================
elif menu == "🤖 Prediksi":
    st.title("🤖 Prediksi Default")

    col1, col2 = st.columns(2)

    with col1:
        LIMIT_BAL = st.number_input("Limit Kredit", 0, 1000000, 200000)
        AGE = st.slider("Usia", 21, 80, 35)
        PAY_0 = st.selectbox("Status Pembayaran Terakhir", [-1, 0, 1, 2])

    with col2:
        BILL_AMT1 = st.number_input("Tagihan Terakhir", 0, 500000, 50000)
        PAY_AMT1 = st.number_input("Pembayaran Terakhir", 0, 500000, 10000)

    if st.button("Prediksi"):
        input_df = pd.DataFrame({
            "LIMIT_BAL": [LIMIT_BAL],
            "AGE": [AGE],
            "PAY_0": [PAY_0],
            "BILL_AMT1": [BILL_AMT1],
            "PAY_AMT1": [PAY_AMT1]
        })

        # Lengkapi fitur yang tidak diinput
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = df[col].mean()

        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ BERISIKO DEFAULT ({prob:.2%})")
        else:
            st.success(f"✅ TIDAK DEFAULT ({1-prob:.2%})")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<center>Credit Card Default Prediction | MLOps Project</center>",
    unsafe_allow_html=True
)
