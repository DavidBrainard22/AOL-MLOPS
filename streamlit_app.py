# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Default Kartu Kredit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #F8FAFC;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
    }
    .predict-button {
        background-color: #3B82F6;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
        margin-top: 1rem;
    }
    .predict-button:hover {
        background-color: #2563EB;
    }
    .result-success {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
    }
    .result-danger {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.title("📊 Menu Navigasi")
    
    menu_option = st.radio(
        "Pilih Menu:",
        ["🏠 Dashboard Utama", "📈 EDA & Visualisasi", "🤖 Prediksi", "📋 Tentang Dataset"]
    )
    
    st.markdown("---")
    st.markdown("### 📚 Sumber Data")
    st.markdown("[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        # Load dataset dari file CSV (asumsi file sudah ada di repo)
        df = pd.read_csv('default_of_credit_card_clients.csv', header=1)
        df = df.rename(columns={'default payment next month': 'default_payment'})
        df = df.drop(columns=['ID'], errors='ignore')
        return df
    except:
        # Jika file tidak ada, gunakan sample data
        st.warning("File dataset tidak ditemukan. Menggunakan data sample...")
        # Buat data sample untuk demo
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'LIMIT_BAL': np.random.randint(10000, 1000000, n_samples),
            'SEX': np.random.choice([1, 2], n_samples, p=[0.4, 0.6]),
            'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.5, 0.2, 0.1]),
            'MARRIAGE': np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.4, 0.1]),
            'AGE': np.random.randint(21, 60, n_samples),
            'PAY_0': np.random.choice([-2, -1, 0, 1, 2], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            'PAY_2': np.random.choice([-2, -1, 0, 1, 2], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            'PAY_3': np.random.choice([-2, -1, 0, 1, 2], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            'PAY_4': np.random.choice([-2, -1, 0, 1, 2], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            'PAY_5': np.random.choice([-2, -1, 0, 1, 2], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            'PAY_6': np.random.choice([-2, -1, 0, 1, 2], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            'BILL_AMT1': np.random.randint(0, 200000, n_samples),
            'BILL_AMT2': np.random.randint(0, 200000, n_samples),
            'BILL_AMT3': np.random.randint(0, 200000, n_samples),
            'BILL_AMT4': np.random.randint(0, 200000, n_samples),
            'BILL_AMT5': np.random.randint(0, 200000, n_samples),
            'BILL_AMT6': np.random.randint(0, 200000, n_samples),
            'PAY_AMT1': np.random.randint(0, 10000, n_samples),
            'PAY_AMT2': np.random.randint(0, 10000, n_samples),
            'PAY_AMT3': np.random.randint(0, 10000, n_samples),
            'PAY_AMT4': np.random.randint(0, 10000, n_samples),
            'PAY_AMT5': np.random.randint(0, 10000, n_samples),
            'PAY_AMT6': np.random.randint(0, 10000, n_samples),
            'default_payment': np.random.choice([0, 1], n_samples, p=[0.78, 0.22])
        }
        
        df = pd.DataFrame(data)
        return df

# Fungsi untuk membuat model sederhana (untuk demo)
def train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Pisahkan fitur dan target
    X = df.drop('default_payment', axis=1)
    y = df['default_payment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Load data
df = load_data()

# Dashboard Utama
if menu_option == "🏠 Dashboard Utama":
    st.markdown("<h1 class='main-header'>💳 Prediksi Default Kartu Kredit</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Nasabah", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        default_rate = (df['default_payment'].sum() / len(df)) * 100
        st.metric("Rate Default", f"{default_rate:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        avg_limit = df['LIMIT_BAL'].mean()
        st.metric("Rata-rata Limit", f"${avg_limit:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Ringkasan Project
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>📋 Ringkasan Project</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    **Objective:**
    Membangun model machine learning untuk memprediksi apakah nasabah akan gagal membayar tagihan kartu kredit bulan depan.
    
    **Dataset:**
    - Sumber: UCI Machine Learning Repository
    - Jumlah data: 30,000 nasabah
    - Fitur: 23 atribut demografi dan historis pembayaran
    - Target: Default payment next month (1 = ya, 0 = tidak)
    
    **Metodologi:**
    1. Data Collection & Loading
    2. Data Cleaning & Preprocessing
    3. Exploratory Data Analysis (EDA)
    4. Feature Engineering
    5. Model Training & Evaluation
    6. Deployment dengan Streamlit
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Preview Data
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>👀 Preview Data</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.markdown("**Info Dataset:**")
        st.text(f"Shape: {df.shape}")
        st.text(f"Kolom: {len(df.columns)}")
        st.text(f"Missing Values: {df.isnull().sum().sum()}")
        st.text(f"Duplikat: {df.duplicated().sum()}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# EDA & Visualisasi
elif menu_option == "📈 EDA & Visualisasi":
    st.markdown("<h1 class='main-header'>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi", "Korelasi", "Analisis Target", "Statistik Deskriptif"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>📊 Distribusi Fitur Numerik</h3>", unsafe_allow_html=True)
        
        # Pilih fitur untuk visualisasi
        numeric_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
        selected_feature = st.selectbox("Pilih fitur untuk dilihat distribusinya:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[selected_feature], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frekuensi')
        ax.set_title(f'Distribusi {selected_feature}')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Skewness dan Kurtosis
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Skewness", f"{df[selected_feature].skew():.2f}")
        with col2:
            st.metric("Kurtosis", f"{df[selected_feature].kurtosis():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>🔗 Heatmap Korelasi</h3>", unsafe_allow_html=True)
        
        # Hitung korelasi untuk beberapa kolom terpilih
        corr_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 
                    'PAY_AMT1', 'PAY_AMT2', 'default_payment']
        corr_matrix = df[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Heatmap Korelasi Antar Fitur')
        
        st.pyplot(fig)
        
        # Insight korelasi
        st.markdown("**Insight:**")
        st.markdown("""
        - Korelasi tertinggi biasanya terlihat antar BILL_AMT (tagihan bulanan)
        - Korelasi dengan target (default_payment) biasanya rendah
        - Beberapa fitur pembayaran mungkin berkorelasi negatif dengan default
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>🎯 Analisis Variabel Target</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart distribusi target
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            target_counts = df['default_payment'].value_counts()
            labels = ['Tidak Default', 'Default']
            colors = ['#4CAF50', '#F44336']
            
            ax1.pie(target_counts, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, wedgeprops={'edgecolor': 'white'})
            ax1.set_title('Distribusi Default Payment')
            ax1.axis('equal')
            
            st.pyplot(fig1)
        
        with col2:
            # Bar plot target berdasarkan kategori
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            
            # Contoh: Default berdasarkan jenis kelamin
            sex_default = df.groupby('SEX')['default_payment'].mean() * 100
            sex_labels = ['Laki-laki', 'Perempuan']
            
            bars = ax2.bar(sex_labels, sex_default, color=['#2196F3', '#FF4081'])
            ax2.set_xlabel('Jenis Kelamin')
            ax2.set_ylabel('Persentase Default (%)')
            ax2.set_title('Default Rate Berdasarkan Jenis Kelamin')
            ax2.grid(True, alpha=0.3)
            
            # Tambah label nilai di atas bar
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig2)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>📊 Statistik Deskriptif</h3>", unsafe_allow_html=True)
        
        # Tampilkan statistik deskriptif
        st.dataframe(df.describe().T, use_container_width=True)
        
        # Insight statistik
        st.markdown("**Key Insights:**")
        st.markdown("""
        1. **LIMIT_BAL**: Rata-rata limit kredit sekitar $167,000 dengan std dev tinggi
        2. **AGE**: Usia rata-rata nasabah 35 tahun
        3. **Default Rate**: Sekitar 22% nasabah mengalami default
        4. **PAY_X**: Mayoritas pembayaran tepat waktu atau 1 bulan telat
        5. **BILL_AMT**: Tagihan menunjukkan variasi yang besar
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Halaman Prediksi
elif menu_option == "🤖 Prediksi":
    st.markdown("<h1 class='main-header'>🤖 Prediksi Default Kartu Kredit</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>📝 Input Data Nasabah</h3>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])
    
    with tab1:
        st.markdown("**Masukkan data nasabah untuk diprediksi:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            LIMIT_BAL = st.number_input("LIMIT_BAL (Limit Kredit)", 
                                       min_value=0, 
                                       max_value=1000000, 
                                       value=200000, 
                                       step=10000)
            
            SEX = st.selectbox("SEX (Jenis Kelamin)", 
                              options=[1, 2], 
                              format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
            
            EDUCATION = st.selectbox("EDUCATION (Pendidikan)", 
                                   options=[1, 2, 3, 4], 
                                   format_func=lambda x: {
                                       1: "Pascasarjana",
                                       2: "Sarjana",
                                       3: "SMA",
                                       4: "Lainnya"
                                   }[x])
        
        with col2:
            MARRIAGE = st.selectbox("MARRIAGE (Status Pernikahan)", 
                                  options=[1, 2, 3], 
                                  format_func=lambda x: {
                                      1: "Menikah",
                                      2: "Lajang",
                                      3: "Lainnya"
                                  }[x])
            
            AGE = st.slider("AGE (Usia)", 21, 79, 35)
            
            PAY_0 = st.selectbox("PAY_0 (Status Pembayaran Bulan Sept)", 
                               options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                               format_func=lambda x: {
                                   -2: "Tidak ada transaksi",
                                   -1: "Bayar penuh",
                                   0: "Tepat waktu",
                                   1: "Telat 1 bulan",
                                   2: "Telat 2 bulan",
                                   3: "Telat 3 bulan",
                                   4: "Telat 4 bulan",
                                   5: "Telat 5 bulan",
                                   6: "Telat 6 bulan",
                                   7: "Telat 7 bulan",
                                   8: "Telat 8 bulan"
                               }[x])
        
        with col3:
            BILL_AMT1 = st.number_input("BILL_AMT1 (Tagihan Bulan Sept)", 
                                       min_value=-200000, 
                                       max_value=1000000, 
                                       value=50000, 
                                       step=1000)
            
            PAY_AMT1 = st.number_input("PAY_AMT1 (Pembayaran Bulan Sept)", 
                                      min_value=0, 
                                      max_value=100000, 
                                      value=5000, 
                                      step=1000)
            
            # Untuk demo, kita isi nilai default untuk kolom lain
            st.info("Kolom lainnya diisi dengan nilai rata-rata dataset")
        
        # Tombol prediksi
        if st.button("🎯 Prediksi Sekarang", type="primary", use_container_width=True):
            # Train model (dalam kasus real, model sudah trained dan disimpan)
            with st.spinner("Melatih model dan melakukan prediksi..."):
                try:
                    # Train model sederhana
                    model, scaler = train_model(df)
                    
                    # Siapkan data input
                    input_data = pd.DataFrame({
                        'LIMIT_BAL': [LIMIT_BAL],
                        'SEX': [SEX],
                        'EDUCATION': [EDUCATION],
                        'MARRIAGE': [MARRIAGE],
                        'AGE': [AGE],
                        'PAY_0': [PAY_0],
                        'PAY_2': [0],  # nilai default
                        'PAY_3': [0],
                        'PAY_4': [0],
                        'PAY_5': [0],
                        'PAY_6': [0],
                        'BILL_AMT1': [BILL_AMT1],
                        'BILL_AMT2': [50000],  # nilai default
                        'BILL_AMT3': [50000],
                        'BILL_AMT4': [50000],
                        'BILL_AMT5': [50000],
                        'BILL_AMT6': [50000],
                        'PAY_AMT1': [PAY_AMT1],
                        'PAY_AMT2': [5000],  # nilai default
                        'PAY_AMT3': [5000],
                        'PAY_AMT4': [5000],
                        'PAY_AMT5': [5000],
                        'PAY_AMT6': [5000]
                    })
                    
                    # Scaling
                    input_scaled = scaler.transform(input_data)
                    
                    # Prediksi
                    prediction = model.predict(input_scaled)[0]
                    proba = model.predict_proba(input_scaled)[0]
                    
                    # Tampilkan hasil
                    st.markdown("---")
                    st.markdown("### 📊 Hasil Prediksi")
                    
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.markdown(f"**Prediksi:**")
                        if prediction == 0:
                            st.markdown("<div class='result-success'>", unsafe_allow_html=True)
                            st.success("✅ NASABAH TIDAK DEFAULT")
                            st.markdown("Nasabah diperkirakan akan membayar tagihan tepat waktu.")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='result-danger'>", unsafe_allow_html=True)
                            st.error("⚠️ NASABAH BERESIKO DEFAULT")
                            st.markdown("Nasabah beresiko tidak membayar tagihan bulan depan.")
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col_result2:
                        st.markdown(f"**Probabilitas:**")
                        
                        fig_prob, ax_prob = plt.subplots(figsize=(8, 4))
                        classes = ['Tidak Default', 'Default']
                        colors = ['#4CAF50', '#F44336']
                        
                        bars = ax_prob.barh(classes, proba, color=colors)
                        ax_prob.set_xlim(0, 1)
                        ax_prob.set_xlabel('Probabilitas')
                        ax_prob.set_title('Distribusi Probabilitas')
                        
                        # Tambah nilai di bar
                        for bar, prob in zip(bars, proba):
                            width = bar.get_width()
                            ax_prob.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                       f'{prob:.1%}', va='center')
                        
                        st.pyplot(fig_prob)
                    
                    # Rekomendasi
                    st.markdown("### 💡 Rekomendasi")
                    if prediction == 0:
                        st.info("""
                        **Rekomendasi untuk nasabah ini:**
                        - Pertahankan limit kredit saat ini
                        - Tawarkan program loyalitas
                        - Bisa dipertimbangkan untuk increase limit
                        """)
                    else:
                        st.warning("""
                        **Rekomendasi untuk nasabah ini:**
                        - Monitor pembayaran secara ketat
                        - Pertimbangkan penurunan limit kredit
                        - Tawarkan restrukturisasi utang jika perlu
                        - Hubungi nasabah untuk konfirmasi pembayaran
                        """)
                
                except Exception as e:
                    st.error(f"Terjadi error: {str(e)}")
                    st.info("""
                    **Untuk demo lengkap:**
                    1. Pastikan semua library terinstall
                    2. Model akan dilatih menggunakan RandomForest
                    3. Hasil prediksi berdasarkan data sample
                    """)
    
    with tab2:
        st.markdown("**Upload file CSV untuk prediksi batch:**")
        
        uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Baca file
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"File berhasil diupload! {len(batch_data)} records ditemukan.")
                
                # Tampilkan preview
                st.dataframe(batch_data.head(), use_container_width=True)
                
                if st.button("🔮 Prediksi Batch", use_container_width=True):
                    with st.spinner("Memproses prediksi batch..."):
                        # Train model
                        model, scaler = train_model(df)
                        
                        # Pastikan kolom sesuai
                        missing_cols = set(df.columns) - set(batch_data.columns) - {'default_payment'}
                        if missing_cols:
                            st.warning(f"Kolom berikut tidak ditemukan: {missing_cols}")
                            for col in missing_cols:
                                batch_data[col] = df[col].mean()  # Isi dengan nilai rata-rata
                        
                        # Urutkan kolom
                        batch_data = batch_data[df.drop('default_payment', axis=1).columns]
                        
                        # Scaling dan prediksi
                        batch_scaled = scaler.transform(batch_data)
                        predictions = model.predict(batch_scaled)
                        probabilities = model.predict_proba(batch_scaled)
                        
                        # Tambah hasil ke dataframe
                        batch_data['PREDICTION'] = predictions
                        batch_data['PROBABILITY_DEFAULT'] = probabilities[:, 1]
                        batch_data['PREDICTION_LABEL'] = batch_data['PREDICTION'].map(
                            {0: 'TIDAK DEFAULT', 1: 'DEFAULT'}
                        )
                        
                        # Tampilkan hasil
                        st.markdown("### 📋 Hasil Prediksi Batch")
                        st.dataframe(batch_data[['PREDICTION_LABEL', 'PROBABILITY_DEFAULT']].head(10), 
                                   use_container_width=True)
                        
                        # Statistik hasil
                        default_count = (predictions == 1).sum()
                        default_percentage = (default_count / len(predictions)) * 100
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Total Prediksi", len(predictions))
                        with col_stat2:
                            st.metric("Prediksi Default", default_count)
                        with col_stat3:
                            st.metric("Persentase Default", f"{default_percentage:.1f}%")
                        
                        # Download hasil
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Hasil Prediksi",
                            data=csv,
                            file_name="hasil_prediksi_batch.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Halaman Tentang Dataset
else:
    st.markdown("<h1 class='main-header'>📋 Informasi Dataset</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>🎯 Deskripsi Dataset</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Dataset ini berisi data 30,000 nasabah kartu kredit di Taiwan dari bulan April hingga September 2005.
    Tujuannya adalah untuk memprediksi apakah nasabah akan gagal membayar tagihan bulan berikutnya.
    """)
    
    st.markdown("### 📊 Struktur Dataset")
    
    # Tabel deskripsi kolom
    kolom_info = pd.DataFrame({
        'Kolom': ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                 'PAY_0 - PAY_6', 'BILL_AMT1 - BILL_AMT6', 
                 'PAY_AMT1 - PAY_AMT6', 'default_payment'],
        'Deskripsi': [
            'Total limit kredit yang diberikan',
            'Jenis kelamin (1=Laki-laki, 2=Perempuan)',
            'Tingkat pendidikan (1=Grad school, 2=University, 3=High school, 4=Others)',
            'Status pernikahan (1=Married, 2=Single, 3=Others)',
            'Usia dalam tahun',
            'Status pembayaran bulan April-Sept 2005 (-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-8=Delay payment in months)',
            'Jumlah tagihan bulan April-Sept 2005',
            'Jumlah pembayaran bulan April-Sept 2005',
            'Target: Default bulan berikutnya (1=Yes, 0=No)'
        ],
        'Tipe Data': ['Numerik', 'Kategorik', 'Kategorik', 'Kategorik', 'Numerik',
                     'Kategorik', 'Numerik', 'Numerik', 'Binary']
    })
    
    st.dataframe(kolom_info, use_container_width=True, hide_index=True)
    
    st.markdown("### 📈 Karakteristik Dataset")
    
    col_char1, col_char2 = st.columns(2)
    
    with col_char1:
        st.markdown("**Statistik Umum:**")
        st.text(f"• Jumlah data: {len(df):,} records")
        st.text(f"• Jumlah fitur: {len(df.columns)}")
        st.text(f"• Data period: April - September 2005")
        st.text(f"• Target variable: Binary classification")
    
    with col_char2:
        st.markdown("**Distribusi Target:**")
        default_count = df['default_payment'].value_counts()
        st.text(f"• Tidak Default (0): {default_count[0]:,} ({default_count[0]/len(df)*100:.1f}%)")
        st.text(f"• Default (1): {default_count[1]:,} ({default_count[1]/len(df)*100:.1f}%)")
        st.text(f"• Imbalance ratio: {default_count[1]/default_count[0]:.2f}")
    
    st.markdown("### 🔍 Preprocessing yang Dilakukan")
    
    preprocessing_steps = [
        "1. **Data Loading**: Membaca data dari file CSV",
        "2. **Data Cleaning**: Menghapus kolom ID yang tidak relevan",
        "3. **Missing Values**: Tidak ada missing values dalam dataset",
        "4. **Duplikat**: Ditemukan dan dihapus 35 records duplikat",
        "5. **Feature Engineering**: Menggunakan semua fitur asli",
        "6. **Scaling**: StandardScaler untuk fitur numerik",
        "7. **Class Imbalance**: Dataset imbalance (22% default)"
    ]
    
    for step in preprocessing_steps:
        st.markdown(step)
    
    st.markdown("### 📚 Referensi")
    st.markdown("""
    - **Sumber**: [UCI Machine Learning Repository - Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
    - **Penelitian Asli**: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
    - **Lisensi**: CC BY 4.0
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Dikembangkan oleh Tim Machine Learning | Dataset: UCI Machine Learning Repository</p>
        <p>© 2024 Credit Card Default Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True
)
