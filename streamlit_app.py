import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Credit Default Prediction",
    layout="wide"
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("Project_MLOps.ipynb")

model = load_model()

# =========================
# UI Header
# =========================
st.title("💳 Credit Default Risk Prediction")
st.write(
    "Aplikasi ini membantu memprediksi risiko gagal bayar berdasarkan "
    "data kredit bulanan. Tidak diperlukan pengetahuan teknis."
)

st.divider()

# =========================
# User Input
# =========================
st.subheader("📌 Informasi Nasabah")

col1, col2, col3 = st.columns(3)

with col1:
    LIMIT_BAL = st.number_input("Limit Kredit", min_value=0)

with col2:
    SEX = st.selectbox("Jenis Kelamin", {1: "Laki-laki", 2: "Perempuan"}.keys(), format_func=lambda x: {1:"Laki-laki",2:"Perempuan"}[x])

with col3:
    AGE = st.number_input("Usia", min_value=18, max_value=100)

EDUCATION = st.selectbox(
    "Pendidikan",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "S2/S3",
        2: "S1",
        3: "SMA",
        4: "Lainnya"
    }[x]
)

MARRIAGE = st.selectbox(
    "Status Pernikahan",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Menikah",
        2: "Lajang",
        3: "Lainnya"
    }[x]
)

st.divider()

# =========================
# Monthly Payment Status
# =========================
st.subheader("📅 Status Pembayaran (6 Bulan Terakhir)")

pay_cols = st.columns(6)
PAY_STATUS = []

for i, col in enumerate(pay_cols):
    with col:
        PAY_STATUS.append(
            st.number_input(f"Bulan {i+1}", min_value=-2, max_value=8, value=0)
        )

st.divider()

# =========================
# Billing Amounts
# =========================
st.subheader("💵 Tagihan Bulanan")

bill_cols = st.columns(6)
BILL_AMT = []

for i, col in enumerate(bill_cols):
    with col:
        BILL_AMT.append(
            st.number_input(f"BILL_AMT{i+1}", min_value=0)
        )

st.divider()

# =========================
# Payment Amounts
# =========================
st.subheader("💰 Pembayaran Bulanan")

pay_amt_cols = st.columns(6)
PAY_AMT = []

for i, col in enumerate(pay_amt_cols):
    with col:
        PAY_AMT.append(
            st.number_input(f"PAY_AMT{i+1}", min_value=0)
        )

st.divider()

# =========================
# Prediction
# =========================
if st.button("🔍 Prediksi Risiko Default"):

    input_data = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,

        "PAY_0": PAY_STATUS[0],
        "PAY_2": PAY_STATUS[1],
        "PAY_3": PAY_STATUS[2],
        "PAY_4": PAY_STATUS[3],
        "PAY_5": PAY_STATUS[4],
        "PAY_6": PAY_STATUS[5],

        "BILL_AMT1": BILL_AMT[0],
        "BILL_AMT2": BILL_AMT[1],
        "BILL_AMT3": BILL_AMT[2],
        "BILL_AMT4": BILL_AMT[3],
        "BILL_AMT5": BILL_AMT[4],
        "BILL_AMT6": BILL_AMT[5],

        "PAY_AMT1": PAY_AMT[0],
        "PAY_AMT2": PAY_AMT[1],
        "PAY_AMT3": PAY_AMT[2],
        "PAY_AMT4": PAY_AMT[3],
        "PAY_AMT5": PAY_AMT[4],
        "PAY_AMT6": PAY_AMT[5],
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ Risiko Gagal Bayar Tinggi\n\nProbabilitas: {probability:.2%}")
    else:
        st.success(f"✅ Risiko Gagal Bayar Rendah\n\nProbabilitas: {probability:.2%}")
