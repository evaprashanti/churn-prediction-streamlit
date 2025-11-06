import streamlit as st
import pandas as pd
import joblib

# Judul aplikasi
st.title("Prediksi Churn Pelanggan Telekomunikasi")

# Load model dan scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Input fitur dari pengguna
st.header("Masukkan Data Pelanggan")

tenure = st.number_input("Lama Berlangganan (bulan)", min_value=0)
monthly_charges = st.number_input("Biaya Bulanan", min_value=0.0)
total_charges = st.number_input("Total Biaya", min_value=0.0)

# Button prediksi
if st.button("Prediksi"):
    # Buat dataframe
    data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Scaling
    data_scaled = scaler.transform(data)

    # Prediksi
    pred = model.predict(data_scaled)

    if pred[0] == 1:
        st.error("⚠️ Pelanggan berpotensi CHURN!")
    else:
        st.success("✅ Pelanggan diprediksi TIDAK churn.")
