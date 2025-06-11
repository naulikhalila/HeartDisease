import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load model dan scaler
model = joblib.load('logistic_regression_heart_model.pkl')
scaler = joblib.load('scaler_heart.pkl')

st.title('Prediksi Risiko Penyakit Jantung')

st.sidebar.title("Navigasi") 
page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi", 
"Hasil Evaluasi"]) 


if page == "Prediksi":
    st.write('Masukkan data pasien di bawah ini:')
    
    # Input data dari user
    age = st.number_input('Umur', min_value=1, max_value=120, value=50)
    trestbps = st.number_input('Tekanan Darah Istirahat (mm Hg)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Kolesterol (mg/dl)', min_value=100, max_value=600, value=200)
    thalach = st.number_input('Detak Jantung Maksimum (bpm)', min_value=60, max_value=220, value=150)
    oldpeak = st.number_input('Depresi ST', min_value=0.0, max_value=6.0, value=1.0, step=0.1)

    # Jika tombol ditekan
    if st.button('Prediksi'):
        # Buat array fitur
        X_new = np.array([[age, trestbps, chol, thalach, oldpeak]])
        
        # Scaling fitur
        X_new_scaled = scaler.transform(X_new)
        
        # Prediksi
        prediction = model.predict(X_new_scaled)
        proba = model.predict_proba(X_new_scaled)[0][1]
        proba_percent = proba * 100

        if prediction == 1:
            st.error(f"Pasien berisiko terkena penyakit jantung! (Probabilitas: {proba_percent:.2f}%)")
        else:
            st.success(f"Pasien tidak berisiko terkena penyakit jantung. (Probabilitas: {proba_percent:.2f}%)")

elif page == "Hasil Evaluasi":
    st.header("Evaluasi Data dan Model")

    # Load data
    df = pd.read_csv('heart_disease_raw.csv')
    features_cluster = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']

    # -----------------------------
    # 1. Clustering Analysis DULU
    # -----------------------------
    st.subheader("🔍 Analisis Clustering (KMeans)")

    df_scaled = df.copy()
    scaler_cluster = StandardScaler()
    df_scaled[features_cluster] = scaler_cluster.fit_transform(df[features_cluster])

    X_cluster = df_scaled[features_cluster]

    # Elbow Method
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_cluster)
        inertias.append(kmeans.inertia_)

    st.write("**Elbow Method - Menentukan Jumlah Cluster**")
    st.markdown("""k=3""")
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, 11), inertias, marker='o')
    ax2.set_xlabel("Jumlah Cluster (k)")
    ax2.set_ylabel("Inertia")
    ax2.set_title("Elbow Method")
    ax2.grid(True)
    st.pyplot(fig2)

    # Clustering dengan k=3
    kmeans_final = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans_final.fit_predict(X_cluster)
    df_scaled['Cluster'] = clusters
    df_with_cluster = df.copy()
    df_with_cluster['Cluster'] = clusters
   

    st.write("**Visualisasi Clustering (Age vs Cholestoral)**")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df_scaled, x='age', y='cholestoral', hue='Cluster', palette='Set2', ax=ax3)
    ax3.set_title("Cluster berdasarkan Umur dan Kolesterol (scaled)")
    ax3.set_xlabel("Age (scaled)")
    ax3.set_ylabel("Cholestoral (scaled)")
    st.pyplot(fig3)

    st.write("**Rangkuman Statistik Tiap Cluster**")
    cluster_summary = df_with_cluster.groupby('Cluster')[features_cluster + ['target']].mean()
    st.dataframe(cluster_summary.style.format("{:.2f}"))

    st.markdown("""
    🔍 Profil Setiap Cluster
Cluster 0 🟢 "Pasien Usia Lanjut Risiko Rendah"

👥 Karakteristik:
Usia: 59.07 tahun (usia lanjut)
Tekanan darah: 126.39 mmHg (normal)
Kolesterol: 229.71 mg/dl (rendah)
Detak jantung maksimum: 127.46 bpm (rendah)
Oldpeak: 1.91 (tinggi - menunjukkan kondisi jantung yang stabil saat istirahat)

🎯 Tingkat Risiko: RENDAH (24%)
💡 Interpretasi: Meskipun berusia lanjut, kelompok ini memiliki profil kardiovaskular yang relatif sehat dengan kolesterol rendah dan tekanan darah normal.

Cluster 1 🔴 "Pasien Muda Risiko Tinggi"

👥 Karakteristik:
Usia: 48.46 tahun (termuda)
Tekanan darah: 125.43 mmHg (normal)
Kolesterol: 233.76 mg/dl (sedang)
Detak jantung maksimum: 163.92 bpm (tinggi)
Oldpeak: 0.40 (rendah - menunjukkan respons jantung yang tidak normal)

🎯 Tingkat Risiko: TINGGI (76%)
💡 Interpretasi: Kelompok ini meski relatif muda, tetapi memiliki risiko tertinggi. Detak jantung maksimum yang tinggi dan oldpeak yang rendah menunjukkan adanya gangguan fungsi jantung.

Cluster 2 🟡 "Pasien Usia Lanjut Risiko Sedang"

👥 Karakteristik:
Usia: 60.24 tahun (tertua)
Tekanan darah: 150.88 mmHg (tinggi)
Kolesterol: 292.06 mg/dl (tinggi)
Detak jantung maksimum: 148.20 bpm (sedang)
Oldpeak: 1.31 (sedang)

🎯 Tingkat Risiko: SEDANG (39%)
💡 Interpretasi: Kelompok ini memiliki faktor risiko klasik (usia lanjut, hipertensi, kolesterol tinggi) namun masih dalam kategori risiko sedang.
    """)


    # -----------------------------
    # 2. Evaluasi Model Klasifikasi
    # -----------------------------
    st.subheader("📌 Evaluasi Model Klasifikasi")

    features_clf = features_cluster
    X = df[features_clf]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.write(f"**Akurasi Model:** {accuracy * 100:.2f}%")

    st.write("**Confusion Matrix:**")
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    st.write("**Classification Report:**")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)