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
    .form-section {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }
    .form-title {
        font-size: 1.25rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3B82F6;
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
        # Load dataset dari file CSV
        df = pd.read_csv('default_of_credit_card_clients.csv', header=1)
        df = df.rename(columns={'default payment next month': 'default_payment'})
        df = df.drop(columns=['ID'], errors='ignore')
        return df
    except:
        # Jika file tidak ada, buat data sample
        st.warning("File dataset tidak ditemukan. Menggunakan data sample...")
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

# Fungsi untuk membuat model
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
        
        numeric_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
        selected_feature = st.selectbox("Pilih fitur untuk dilihat distribusinya:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[selected_feature], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frekuensi')
        ax.set_title(f'Distribusi {selected_feature}')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Skewness", f"{df[selected_feature].skew():.2f}")
        with col2:
            st.metric("Kurtosis", f"{df[selected_feature].kurtosis():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>🔗 Heatmap Korelasi</h3>", unsafe_allow_html=True)
        
        corr_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 
                    'PAY_AMT1', 'PAY_AMT2', 'default_payment']
        corr_matrix = df[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Heatmap Korelasi Antar Fitur')
        
        st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>🎯 Analisis Variabel Target</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
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
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sex_default = df.groupby('SEX')['default_payment'].mean() * 100
            sex_labels = ['Laki-laki', 'Perempuan']
            
            bars = ax2.bar(sex_labels, sex_default, color=['#2196F3', '#FF4081'])
            ax2.set_xlabel('Jenis Kelamin')
            ax2.set_ylabel('Persentase Default (%)')
            ax2.set_title('Default Rate Berdasarkan Jenis Kelamin')
            ax2.grid(True, alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig2)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>📊 Statistik Deskriptif</h3>", unsafe_allow_html=True)
        
        st.dataframe(df.describe().T, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Halaman Prediksi - INPUT LENGKAP
elif menu_option == "🤖 Prediksi":
    st.markdown("<h1 class='main-header'>🤖 Prediksi Default Kartu Kredit</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>📝 Input Data Nasabah (Semua Fitur)</h3>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Input Manual Lengkap", "Upload CSV"])
    
    with tab1:
        st.markdown("**Masukkan SEMUA data nasabah untuk prediksi yang akurat:**")
        
        # Buat expander untuk setiap kelompok fitur
        with st.expander("📋 **1. Data Demografis**", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                LIMIT_BAL = st.number_input("LIMIT_BAL (Limit Kredit dalam $)", 
                                          min_value=0, 
                                          max_value=1000000, 
                                          value=200000, 
                                          step=10000,
                                          help="Total jumlah kredit yang diberikan")
                
                SEX = st.selectbox("SEX (Jenis Kelamin)", 
                                  options=[1, 2], 
                                  format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan",
                                  help="1 = Laki-laki, 2 = Perempuan")
            
            with col2:
                EDUCATION = st.selectbox("EDUCATION (Pendidikan)", 
                                       options=[1, 2, 3, 4], 
                                       format_func=lambda x: {
                                           1: "Pascasarjana",
                                           2: "Sarjana",
                                           3: "SMA",
                                           4: "Lainnya"
                                       }[x],
                                       help="1 = Pascasarjana, 2 = Sarjana, 3 = SMA, 4 = Lainnya")
                
                MARRIAGE = st.selectbox("MARRIAGE (Status Pernikahan)", 
                                      options=[1, 2, 3], 
                                      format_func=lambda x: {
                                          1: "Menikah",
                                          2: "Lajang",
                                          3: "Lainnya"
                                      }[x],
                                      help="1 = Menikah, 2 = Lajang, 3 = Lainnya")
            
            with col3:
                AGE = st.slider("AGE (Usia dalam tahun)", 21, 79, 35)
        
        with st.expander("📅 **2. Status Pembayaran (PAY_X)**"):
            st.markdown("**Status pembayaran bulan sebelumnya:**")
            col1, col2 = st.columns(2)
            
            pay_options = {
                -2: "Tidak ada transaksi/konsumsi",
                -1: "Bayar penuh",
                0: "Pembayaran minimum/tidak ada keterlambatan",
                1: "Telat 1 bulan",
                2: "Telat 2 bulan",
                3: "Telat 3 bulan",
                4: "Telat 4 bulan",
                5: "Telat 5 bulan",
                6: "Telat 6 bulan",
                7: "Telat 7 bulan",
                8: "Telat 8 bulan"
            }
            
            with col1:
                PAY_0 = st.selectbox("PAY_0 (Status Sept 2005)", 
                                   options=list(pay_options.keys()),
                                   format_func=lambda x: pay_options[x])
                
                PAY_2 = st.selectbox("PAY_2 (Status Agustus 2005)", 
                                   options=list(pay_options.keys()),
                                   format_func=lambda x: pay_options[x])
                
                PAY_4 = st.selectbox("PAY_4 (Status Juni 2005)", 
                                   options=list(pay_options.keys()),
                                   format_func=lambda x: pay_options[x])
            
            with col2:
                PAY_3 = st.selectbox("PAY_3 (Status Juli 2005)", 
                                   options=list(pay_options.keys()),
                                   format_func=lambda x: pay_options[x])
                
                PAY_5 = st.selectbox("PAY_5 (Status Mei 2005)", 
                                   options=list(pay_options.keys()),
                                   format_func=lambda x: pay_options[x])
                
                PAY_6 = st.selectbox("PAY_6 (Status April 2005)", 
                                   options=list(pay_options.keys()),
                                   format_func=lambda x: pay_options[x])
        
        with st.expander("💰 **3. Jumlah Tagihan (BILL_AMT)**"):
            st.markdown("**Jumlah tagihan bulan sebelumnya (dalam $):**")
            col1, col2 = st.columns(2)
            
            with col1:
                BILL_AMT1 = st.number_input("BILL_AMT1 (Tagihan Sept 2005)", 
                                          min_value=-200000, 
                                          max_value=1000000, 
                                          value=50000, 
                                          step=1000)
                
                BILL_AMT2 = st.number_input("BILL_AMT2 (Tagihan Agustus 2005)", 
                                          min_value=-200000, 
                                          max_value=1000000, 
                                          value=45000, 
                                          step=1000)
                
                BILL_AMT3 = st.number_input("BILL_AMT3 (Tagihan Juli 2005)", 
                                          min_value=-200000, 
                                          max_value=1000000, 
                                          value=40000, 
                                          step=1000)
            
            with col2:
                BILL_AMT4 = st.number_input("BILL_AMT4 (Tagihan Juni 2005)", 
                                          min_value=-200000, 
                                          max_value=1000000, 
                                          value=35000, 
                                          step=1000)
                
                BILL_AMT5 = st.number_input("BILL_AMT5 (Tagihan Mei 2005)", 
                                          min_value=-200000, 
                                          max_value=1000000, 
                                          value=30000, 
                                          step=1000)
                
                BILL_AMT6 = st.number_input("BILL_AMT6 (Tagihan April 2005)", 
                                          min_value=-200000, 
                                          max_value=1000000, 
                                          value=25000, 
                                          step=1000)
        
        with st.expander("💵 **4. Jumlah Pembayaran (PAY_AMT)**"):
            st.markdown("**Jumlah pembayaran bulan sebelumnya (dalam $):**")
            col1, col2 = st.columns(2)
            
            with col1:
                PAY_AMT1 = st.number_input("PAY_AMT1 (Pembayaran Sept 2005)", 
                                         min_value=0, 
                                         max_value=100000, 
                                         value=5000, 
                                         step=1000)
                
                PAY_AMT2 = st.number_input("PAY_AMT2 (Pembayaran Agustus 2005)", 
                                         min_value=0, 
                                         max_value=100000, 
                                         value=4500, 
                                         step=1000)
                
                PAY_AMT3 = st.number_input("PAY_AMT3 (Pembayaran Juli 2005)", 
                                         min_value=0, 
                                         max_value=100000, 
                                         value=4000, 
                                         step=1000)
            
            with col2:
                PAY_AMT4 = st.number_input("PAY_AMT4 (Pembayaran Juni 2005)", 
                                         min_value=0, 
                                         max_value=100000, 
                                         value=3500, 
                                         step=1000)
                
                PAY_AMT5 = st.number_input("PAY_AMT5 (Pembayaran Mei 2005)", 
                                         min_value=0, 
                                         max_value=100000, 
                                         value=3000, 
                                         step=1000)
                
                PAY_AMT6 = st.number_input("PAY_AMT6 (Pembayaran April 2005)", 
                                         min_value=0, 
                                         max_value=100000, 
                                         value=2500, 
                                         step=1000)
        
        # Tombol prediksi
        if st.button("🎯 Prediksi Sekarang", type="primary", use_container_width=True):
            with st.spinner("Melatih model dan melakukan prediksi..."):
                try:
                    # Train model
                    model, scaler = train_model(df)
                    
                    # Siapkan data input LENGKAP
                    input_data = pd.DataFrame({
                        'LIMIT_BAL': [LIMIT_BAL],
                        'SEX': [SEX],
                        'EDUCATION': [EDUCATION],
                        'MARRIAGE': [MARRIAGE],
                        'AGE': [AGE],
                        'PAY_0': [PAY_0],
                        'PAY_2': [PAY_2],
                        'PAY_3': [PAY_3],
                        'PAY_4': [PAY_4],
                        'PAY_5': [PAY_5],
                        'PAY_6': [PAY_6],
                        'BILL_AMT1': [BILL_AMT1],
                        'BILL_AMT2': [BILL_AMT2],
                        'BILL_AMT3': [BILL_AMT3],
                        'BILL_AMT4': [BILL_AMT4],
                        'BILL_AMT5': [BILL_AMT5],
                        'BILL_AMT6': [BILL_AMT6],
                        'PAY_AMT1': [PAY_AMT1],
                        'PAY_AMT2': [PAY_AMT2],
                        'PAY_AMT3': [PAY_AMT3],
                        'PAY_AMT4': [PAY_AMT4],
                        'PAY_AMT5': [PAY_AMT5],
                        'PAY_AMT6': [PAY_AMT6]
                    })
                    
                    # Tampilkan data input
                    st.markdown("---")
                    st.markdown("### 📋 Data Input yang Dimasukkan")
                    st.dataframe(input_data.T.rename(columns={0: 'Nilai'}), use_container_width=True)
                    
                    # Scaling
                    input_scaled = scaler.transform(input_data)
                    
                    # Prediksi
                    prediction = model.predict(input_scaled)[0]
                    proba = model.predict_proba(input_scaled)[0]
                    
                    # Tampilkan hasil
                    st.markdown("### 📊 Hasil Prediksi")
                    
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.markdown(f"**Prediksi:**")
                        if prediction == 0:
                            st.markdown("<div class='result-success'>", unsafe_allow_html=True)
                            st.success("✅ NASABAH TIDAK DEFAULT")
                            st.markdown("**Probabilitas Tidak Default:** {:.1%}".format(proba[0]))
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='result-danger'>", unsafe_allow_html=True)
                            st.error("⚠️ NASABAH BERESIKO DEFAULT")
                            st.markdown("**Probabilitas Default:** {:.1%}".format(proba[1]))
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col_result2:
                        st.markdown(f"**Probabilitas Detail:**")
                        
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
                        - ✅ Nasabah menunjukkan profil pembayaran yang sehat
                        - ✅ Pertahankan limit kredit saat ini
                        - ✅ Tawarkan program loyalitas atau rewards
                        - ✅ Pertimbangkan untuk meningkatkan limit kredit
                        - ✅ Monitor berkala untuk deteksi dini perubahan perilaku
                        """)
                    else:
                        st.warning("""
                        **Rekomendasi untuk nasabah ini:**
                        - ⚠️ Tingkatkan monitoring pembayaran
                        - ⚠️ Pertimbangkan penurunan limit kredit
                        - ⚠️ Hubungi nasabah untuk konfirmasi pembayaran
                        - ⚠️ Tawarkan opsi restrukturisasi utang jika perlu
                        - ⚠️ Flag nasabah untuk review lebih mendalam
                        - ⚠️ Pertimbangkan biaya keterlambatan yang lebih tinggi
                        """)
                    
                    # Feature Importance
                    st.markdown("### 🔍 Kontribusi Fitur (Top 10)")
                    
                    # Get feature importance
                    feature_importance = pd.DataFrame({
                        'feature': df.drop('default_payment', axis=1).columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    bars_imp = ax_imp.barh(feature_importance['feature'], 
                                          feature_importance['importance'],
                                          color='#3B82F6')
                    ax_imp.set_xlabel('Importance Score')
                    ax_imp.set_title('Top 10 Fitur Paling Penting')
                    ax_imp.invert_yaxis()
                    
                    for bar in bars_imp:
                        width = bar.get_width()
                        ax_imp.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                                  f'{width:.3f}', va='center')
                    
                    st.pyplot(fig_imp)
                
                except Exception as e:
                    st.error(f"Terjadi error: {str(e)}")
    
    with tab2:
        st.markdown("**Upload file CSV untuk prediksi batch:**")
        
        uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"File berhasil diupload! {len(batch_data)} records ditemukan.")
                
                # Cek kolom
                required_cols = list(df.drop('default_payment', axis=1).columns)
                missing_cols = set(required_cols) - set(batch_data.columns)
                
                if missing_cols:
                    st.error(f"File CSV harus memiliki semua kolom berikut: {missing_cols}")
                else:
                    st.dataframe(batch_data.head(), use_container_width=True)
                    
                    if st.button("🔮 Prediksi Batch", use_container_width=True):
                        with st.spinner("Memproses prediksi batch..."):
                            model, scaler = train_model(df)
                            
                            # Urutkan kolom
                            batch_data = batch_data[required_cols]
                            
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
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>© 2024 Credit Card Default Prediction System | Dataset: UCI Machine Learning Repository</p>
    </div>
    """,
    unsafe_allow_html=True
)
