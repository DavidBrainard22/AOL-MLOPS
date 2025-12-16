import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Credit Default Prediction",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction")
st.write(
    "Aplikasi ini membantu memprediksi risiko gagal bayar kartu kredit "
    "berdasarkan riwayat pembayaran dan tagihan 6 bulan terakhir."
)

# ===============================
# LOAD & TRAIN MODEL
# ===============================
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("credit_default.csv")  # pastikan nama sama di repo

    # Target
    X = df.drop(columns=["default.payment.next.month", "ID"])
    y = df["default.payment.next.month"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

model = train_model()

# ===============================
# USER INPUT
# ===============================
st.header("📥 Masukkan Data Nasabah")

col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input(
        "Limit Kredit (NT$)",
        min_value=0,
        step=1000
    )
    age = st.number_input("Umur", min_value=18, max_value=100)

with col2:
    sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    education = st.selectbox(
        "Pendidikan",
        ["SMA", "Universitas", "Pascasarjana", "Lainnya"]
    )

with col3:
    marriage = st.selectbox(
        "Status Pernikahan",
        ["Belum Menikah", "Menikah", "Lainnya"]
    )

st.subheader("📊 Riwayat Tagihan 6 Bulan Terakhir")

bill_amts = []
pay_amts = []
pay_status = []

for i in range(1, 7):
    with st.expander(f"Bulan ke-{i}"):
        bill_amts.append(st.number_input(
            f"Tagihan Bulan {i} (BILL_AMT{i})", step=100
        ))
        pay_amts.append(st.number_input(
            f"Pembayaran Bulan {i} (PAY_AMT{i})", step=100
        ))
        pay_status.append(st.number_input(
            f"Status Pembayaran Bulan {i} (PAY_{i})",
            min_value=-2, max_value=8
        ))

# ===============================
# ENCODING
# ===============================
sex = 1 if sex == "Laki-laki" else 2
edu_map = {
    "SMA": 2,
    "Universitas": 1,
    "Pascasarjana": 0,
    "Lainnya": 3
}
marriage_map = {
    "Belum Menikah": 1,
    "Menikah": 2,
    "Lainnya": 3
}

education = edu_map[education]
marriage = marriage_map[marriage]

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Prediksi Risiko"):
    input_data = np.array([[
        limit_bal, sex, education, marriage, age,
        *pay_status,
        *bill_amts,
        *pay_amts
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Hasil Prediksi")

    if prediction == 1:
        st.error(
            f"⚠️ Risiko Gagal Bayar TINGGI\n\nProbabilitas: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Risiko Gagal Bayar RENDAH\n\nProbabilitas: {probability:.2%}"
        )

    st.caption(
        "Catatan: Prediksi ini berbasis pola historis dan tidak menggantikan "
        "keputusan finansial profesional."
    )
