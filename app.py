import streamlit as st
import pandas as pd
import joblib

# Pengaturan tampilan
st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="centered")

st.title("📊 Prediksi Churn Pelanggan (Skala 0.00 – 1.00)")
st.write("Masukkan data pelanggan dalam skala **0.00 – 1.00** (hasil dari penskalaan MinMaxScaler).")

# Load model & scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Input pengguna
st.header("Masukkan Data Pelanggan")
tenure = st.number_input("Lama Berlangganan (0.00–1.00)", min_value=0.00, max_value=1.00, value=0.00, step=0.01, format="%.2f")
monthly_charges = st.number_input("Biaya Bulanan (0.00–1.00)", min_value=0.00, max_value=1.00, value=0.00, step=0.01, format="%.2f")
total_charges = st.number_input("Total Biaya (0.00–1.00)", min_value=0.00, max_value=1.00, value=0.00, step=0.01, format="%.2f")

# Tombol prediksi
if st.button("Prediksi"):
    # Buat DataFrame dari input
    data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Karena input sudah dalam skala 0–1, tidak perlu transform
    pred = model.predict(data)
    proba = model.predict_proba(data)[0][1]

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    if pred[0] == 1:
        st.error(f"⚠️ Pelanggan berpotensi **CHURN** (Probabilitas: {proba*100:.2f}%)")
    else:
        st.success(f"✅ Pelanggan diprediksi **TIDAK CHURN** (Probabilitas: {(1-proba)*100:.2f}%)")

    # Info tambahan di bawah hasil
    st.caption("Keterangan: Model menggunakan Logistic Regression dengan data hasil penskalaan MinMaxScaler (0.00–1.00).")
