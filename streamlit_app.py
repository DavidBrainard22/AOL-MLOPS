import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ux_utils as ux

# Set page config
ux.set_page_config()

# Inject custom CSS
ux.inject_custom_css()

# Header
ux.render_header()

# Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model tidak ditemukan! Harap jalankan `python train_model.py` terlebih dahulu.")
        return None

model = load_model()

if model:
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/893/893097.png", width=100)
    st.sidebar.title("Navigasi")
    st.sidebar.info("Aplikasi ini menggunakan model Random Forest untuk memprediksi kemungkinan gagal bayar kartu kredit.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            ux.render_feature_section("Data Diri", "👤")
            limit_bal = st.number_input("Limit Kartu Kredit (IDR)", min_value=10000, max_value=1000000000, step=100000, value=5000000)
            age = st.number_input("Usia", min_value=17, max_value=100, step=1, value=25)
            
            sex_options = {1: "Laki-laki", 2: "Perempuan"}
            sex = st.selectbox("Jenis Kelamin", options=list(sex_options.keys()), format_func=lambda x: sex_options[x])
            
            edu_options = {1: "Pascasarjana", 2: "Sarjana", 3: "SMA", 4: "Lainnya", 5: "Tidak Diketahui", 6: "Tidak Diketahui"}
            education = st.selectbox("Pendidikan", options=list(edu_options.keys()), format_func=lambda x: edu_options[x], )
            
            mar_options = {1: "Menikah", 2: "Lajang", 3: "Lainnya"}
            marriage = st.selectbox("Status Pernikahan", options=list(mar_options.keys()), format_func=lambda x: mar_options[x])

        with col2:
            ux.render_feature_section("Status Pembayaran (6 Bulan Terakhir)", "📅")
            st.caption("-1: Tepat Waktu, 1-8: Telat n Bulan")
            
            pay_options = {-2: "Tidak ada tagihan", -1: "Lunas", 0: "Bayar Minimum", 1: "Telat 1 Bulan", 2: "Telat 2 Bulan", 3: "Telat 3 Bulan", 4: "Telat 4 Bulan", 5: "Telat 5 Bulan", 6: "Telat 6 Bulan", 7: "Telat 7 Bulan", 8: "Telat 8 Bulan"}
            
            pay_0 = st.selectbox("Status Bulan Ini (September)", options=list(pay_options.keys()), format_func=lambda x: pay_options[x], index=2)
            pay_2 = st.selectbox("Status Bulan Lalu (Agustus)", options=list(pay_options.keys()), format_func=lambda x: pay_options[x], index=2)
            pay_3 = st.selectbox("Status 2 Bulan Lalu (Juli)", options=list(pay_options.keys()), format_func=lambda x: pay_options[x], index=2)
            pay_4 = st.selectbox("Status 3 Bulan Lalu (Juni)", options=list(pay_options.keys()), format_func=lambda x: pay_options[x], index=2)
            pay_5 = st.selectbox("Status 4 Bulan Lalu (Mei)", options=list(pay_options.keys()), format_func=lambda x: pay_options[x], index=2)
            pay_6 = st.selectbox("Status 5 Bulan Lalu (April)", options=list(pay_options.keys()), format_func=lambda x: pay_options[x], index=2)

        st.markdown("<br>", unsafe_allow_html=True)
        ux.render_feature_section("Riwayat Tagihan & Pembayaran", "💰")
        
        # Using tabs for cleaner UI
        tab1, tab2 = st.tabs(["Tagihan (Bill)", "Pembayaran (Pay)"])
        
        with tab1:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                 bill_amt1 = st.number_input("Tagihan Sept (IDR)", step=10000.0)
                 bill_amt2 = st.number_input("Tagihan Agust (IDR)", step=10000.0)
                 bill_amt3 = st.number_input("Tagihan Juli (IDR)", step=10000.0)
            with col_b2:
                 bill_amt4 = st.number_input("Tagihan Juni (IDR)", step=10000.0)
                 bill_amt5 = st.number_input("Tagihan Mei (IDR)", step=10000.0)
                 bill_amt6 = st.number_input("Tagihan April (IDR)", step=10000.0)

        with tab2:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                 pay_amt1 = st.number_input("Bayar Sept (IDR)", step=10000.0)
                 pay_amt2 = st.number_input("Bayar Agust (IDR)", step=10000.0)
                 pay_amt3 = st.number_input("Bayar Juli (IDR)", step=10000.0)
            with col_p2:
                 pay_amt4 = st.number_input("Bayar Juni (IDR)", step=10000.0)
                 pay_amt5 = st.number_input("Bayar Mei (IDR)", step=10000.0)
                 pay_amt6 = st.number_input("Bayar April (IDR)", step=10000.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analisis Resiko")

    if submitted:
        # Data Preparation
        input_data = {
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
        }
        
        df_input = pd.DataFrame(input_data)
        
        # Feature Engineering (Must match train_model.py)
        df_input['TOTAL_BILL_AMT'] = df_input[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
        df_input['TOTAL_PAY_AMT'] = df_input[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)
        df_input['PAYMENT_RATIO'] = df_input['TOTAL_PAY_AMT'] / np.where(df_input['TOTAL_BILL_AMT'] == 0, 1, df_input['TOTAL_BILL_AMT'])
        
        pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        df_input['HAS_MISSED_PAYMENT'] = (df_input[pay_cols] > 0).any(axis=1).astype(int)

        # Prediction
        try:
            prediction = model.predict(df_input)[0]
            probability = model.predict_proba(df_input)[0][1]

            st.markdown("---")
            st.subheader("Hasil Analisis")

            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("Probabilitas Default", f"{probability:.1%}")
            
            with col_res2:
                if prediction == 1:
                    st.markdown("""
                        <div class="prediction-danger">
                            <h3>🚨 Berisiko Gagal Bayar</h3>
                            <p>Nasabah ini memiliki indikator risiko tinggi. Disarankan untuk meninjau ulang atau menolak pengajuan kredit tambahan.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-success">
                            <h3>✅ Aman (Layak Kredit)</h3>
                            <p>Nasabah ini menunjukkan indikator keuangan yang sehat. Risiko gagal bayar rendah.</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
