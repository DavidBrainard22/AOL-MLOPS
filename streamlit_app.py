import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Judul aplikasi
st.title("Prediksi Default Kartu Kredit")
st.write("Aplikasi ini memprediksi apakah nasabah akan gagal membayar tagihan kartu kredit bulan depan.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel('default_of_credit_card_clients.xlsx', header=1)
    df.drop(columns=['ID'], inplace=True)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# Pisahkan fitur dan target
X = df.drop(columns=['default payment next month'])
y = df['default payment next month']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definisikan kolom numerik dan kategorikal
numerical_columns = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
                     'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
                     'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 
                       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# Buat pipeline preprocessing
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numerical_columns),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
])

# Buat pipeline model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# Latih model
model_pipeline.fit(X_train, y_train)

# Simpan model (opsional, untuk penggunaan berikutnya)
joblib.dump(model_pipeline, 'credit_default_model.pkl')

# Sidebar untuk input pengguna
st.sidebar.header("Masukkan Data Nasabah")

# Input untuk setiap fitur
limit_bal = st.sidebar.number_input("LIMIT_BAL (Limit Kredit dalam NT$)", min_value=0, value=200000)
sex = st.sidebar.selectbox("Jenis Kelamin", options=[1, 2], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
education = st.sidebar.selectbox("Pendidikan", options=[1, 2, 3, 4], 
                                 format_func=lambda x: {1: "Pascasarjana", 2: "Sarjana", 3: "SMA", 4: "Lainnya"}.get(x, "Lainnya"))
marriage = st.sidebar.selectbox("Status Pernikahan", options=[1, 2, 3], 
                                format_func=lambda x: {1: "Menikah", 2: "Lajang", 3: "Lainnya"}.get(x, "Lainnya"))
age = st.sidebar.number_input("Usia", min_value=21, max_value=100, value=30)

st.sidebar.subheader("Riwayat Pembayaran (dalam bulan)")
pay_0 = st.sidebar.selectbox("PAY_0 (Status Pembayaran Sept 2005)", 
                             options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                             format_func=lambda x: {
                                 -2: "Tidak ada transaksi", -1: "Bayar penuh", 0: "Bayar minimum", 
                                 1: "Tunggak 1 bulan", 2: "Tunggak 2 bulan", 3: "Tunggak 3 bulan",
                                 4: "Tunggak 4 bulan", 5: "Tunggak 5 bulan", 6: "Tunggak 6 bulan",
                                 7: "Tunggak 7 bulan", 8: "Tunggak 8 bulan"
                             }.get(x, f"Tunggak {x} bulan"))
pay_2 = st.sidebar.selectbox("PAY_2 (Status Pembayaran Agust 2005)", 
                             options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                             format_func=lambda x: {
                                 -2: "Tidak ada transaksi", -1: "Bayar penuh", 0: "Bayar minimum", 
                                 1: "Tunggak 1 bulan", 2: "Tunggak 2 bulan", 3: "Tunggak 3 bulan",
                                 4: "Tunggak 4 bulan", 5: "Tunggak 5 bulan", 6: "Tunggak 6 bulan",
                                 7: "Tunggak 7 bulan", 8: "Tunggak 8 bulan"
                             }.get(x, f"Tunggak {x} bulan"))
pay_3 = st.sidebar.selectbox("PAY_3 (Status Pembayaran Juli 2005)", 
                             options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                             format_func=lambda x: {
                                 -2: "Tidak ada transaksi", -1: "Bayar penuh", 0: "Bayar minimum", 
                                 1: "Tunggak 1 bulan", 2: "Tunggak 2 bulan", 3: "Tunggak 3 bulan",
                                 4: "Tunggak 4 bulan", 5: "Tunggak 5 bulan", 6: "Tunggak 6 bulan",
                                 7: "Tunggak 7 bulan", 8: "Tunggak 8 bulan"
                             }.get(x, f"Tunggak {x} bulan"))
pay_4 = st.sidebar.selectbox("PAY_4 (Status Pembayaran Juni 2005)", 
                             options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                             format_func=lambda x: {
                                 -2: "Tidak ada transaksi", -1: "Bayar penuh", 0: "Bayar minimum", 
                                 1: "Tunggak 1 bulan", 2: "Tunggak 2 bulan", 3: "Tunggak 3 bulan",
                                 4: "Tunggak 4 bulan", 5: "Tunggak 5 bulan", 6: "Tunggak 6 bulan",
                                 7: "Tunggak 7 bulan", 8: "Tunggak 8 bulan"
                             }.get(x, f"Tunggak {x} bulan"))
pay_5 = st.sidebar.selectbox("PAY_5 (Status Pembayaran Mei 2005)", 
                             options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                             format_func=lambda x: {
                                 -2: "Tidak ada transaksi", -1: "Bayar penuh", 0: "Bayar minimum", 
                                 1: "Tunggak 1 bulan", 2: "Tunggak 2 bulan", 3: "Tunggak 3 bulan",
                                 4: "Tunggak 4 bulan", 5: "Tunggak 5 bulan", 6: "Tunggak 6 bulan",
                                 7: "Tunggak 7 bulan", 8: "Tunggak 8 bulan"
                             }.get(x, f"Tunggak {x} bulan"))
pay_6 = st.sidebar.selectbox("PAY_6 (Status Pembayaran April 2005)", 
                             options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                             format_func=lambda x: {
                                 -2: "Tidak ada transaksi", -1: "Bayar penuh", 0: "Bayar minimum", 
                                 1: "Tunggak 1 bulan", 2: "Tunggak 2 bulan", 3: "Tunggak 3 bulan",
                                 4: "Tunggak 4 bulan", 5: "Tunggak 5 bulan", 6: "Tunggak 6 bulan",
                                 7: "Tunggak 7 bulan", 8: "Tunggak 8 bulan"
                             }.get(x, f"Tunggak {x} bulan"))

st.sidebar.subheader("Jumlah Tagihan (dalam NT$)")
bill_amt1 = st.sidebar.number_input("BILL_AMT1 (Tagihan Sept 2005)", min_value=-200000, value=50000)
bill_amt2 = st.sidebar.number_input("BILL_AMT2 (Tagihan Agust 2005)", min_value=-200000, value=50000)
bill_amt3 = st.sidebar.number_input("BILL_AMT3 (Tagihan Juli 2005)", min_value=-200000, value=50000)
bill_amt4 = st.sidebar.number_input("BILL_AMT4 (Tagihan Juni 2005)", min_value=-200000, value=50000)
bill_amt5 = st.sidebar.number_input("BILL_AMT5 (Tagihan Mei 2005)", min_value=-200000, value=50000)
bill_amt6 = st.sidebar.number_input("BILL_AMT6 (Tagihan April 2005)", min_value=-200000, value=50000)

st.sidebar.subheader("Jumlah Pembayaran (dalam NT$)")
pay_amt1 = st.sidebar.number_input("PAY_AMT1 (Pembayaran Sept 2005)", min_value=0, value=5000)
pay_amt2 = st.sidebar.number_input("PAY_AMT2 (Pembayaran Agust 2005)", min_value=0, value=5000)
pay_amt3 = st.sidebar.number_input("PAY_AMT3 (Pembayaran Juli 2005)", min_value=0, value=5000)
pay_amt4 = st.sidebar.number_input("PAY_AMT4 (Pembayaran Juni 2005)", min_value=0, value=5000)
pay_amt5 = st.sidebar.number_input("PAY_AMT5 (Pembayaran Mei 2005)", min_value=0, value=5000)
pay_amt6 = st.sidebar.number_input("PAY_AMT6 (Pembayaran April 2005)", min_value=0, value=5000)

# Buat DataFrame dari input pengguna
input_data = pd.DataFrame({
    'LIMIT_BAL': [limit_bal],
    'SEX': [sex],
    'EDUCATION': [education],
    'MARRIAGE': [marriage],
    'AGE': [age],
    'PAY_0': [pay_0],
    'PAY_2': [pay_2],
    'PAY_3': [pay_3],
    'PAY_4': [pay_4],
    'PAY_5': [pay_5],
    'PAY_6': [pay_6],
    'BILL_AMT1': [bill_amt1],
    'BILL_AMT2': [bill_amt2],
    'BILL_AMT3': [bill_amt3],
    'BILL_AMT4': [bill_amt4],
    'BILL_AMT5': [bill_amt5],
    'BILL_AMT6': [bill_amt6],
    'PAY_AMT1': [pay_amt1],
    'PAY_AMT2': [pay_amt2],
    'PAY_AMT3': [pay_amt3],
    'PAY_AMT4': [pay_amt4],
    'PAY_AMT5': [pay_amt5],
    'PAY_AMT6': [pay_amt6]
})

# Tombol prediksi
if st.sidebar.button("Prediksi"):
    # Lakukan prediksi
    prediction = model_pipeline.predict(input_data)
    prediction_proba = model_pipeline.predict_proba(input_data)
    
    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    
    if prediction[0] == 1:
        st.error(f"**Prediksi: GAGAL BAYAR**")
        st.write(f"Kemungkinan gagal bayar: {prediction_proba[0][1]*100:.2f}%")
        st.write(f"Kemungkinan tidak gagal bayar: {prediction_proba[0][0]*100:.2f}%")
        st.warning("⚠️ Nasabah diprediksi akan mengalami kesulitan dalam pembayaran bulan depan. Disarankan untuk melakukan tindakan preventif.")
    else:
        st.success(f"**Prediksi: TIDAK GAGAL BAYAR**")
        st.write(f"Kemungkinan tidak gagal bayar: {prediction_proba[0][0]*100:.2f}%")
        st.write(f"Kemungkinan gagal bayar: {prediction_proba[0][1]*100:.2f}%")
        st.info("✅ Nasabah diprediksi akan membayar tagihan tepat waktu.")
    
    # Tambahkan penjelasan singkat
    with st.expander("ℹ️ Interpretasi Hasil"):
        st.write("""
        - **TIDAK GAGAL BAYAR (0)**: Nasabah diprediksi akan membayar tagihan bulan depan tepat waktu.
        - **GAGAL BAYAR (1)**: Nasabah diprediksi akan mengalami keterlambatan atau gagal membayar tagihan bulan depan.
        
        Hasil prediksi didasarkan pada data historis pembayaran dan karakteristik nasabah.
        """)

# Tampilkan data input untuk review
with st.expander("📋 Review Data yang Dimasukkan"):
    st.write(input_data)

# Informasi tambahan
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Panduan Penggunaan:**
    1. Isi semua field di sisi kiri.
    2. Klik tombol **Prediksi** untuk melihat hasil.
    3. Hasil akan ditampilkan di panel utama.
    
    **Catatan:** Model dilatih dengan data dari Taiwan (2005).
    """
)

# Footer
st.markdown("---")
st.caption("Aplikasi Prediksi Default Kartu Kredit | Menggunakan model Random Forest")
