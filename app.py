import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="centered")
st.title("📊 Prediksi Churn Pelanggan (Skala 0.00–1.00)")

# Load model dan scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.header("Masukkan Data Pelanggan")
tenure = st.number_input("Lama Berlangganan (0.00–1.00)", min_value=0.00, max_value=1.00, value=0.00, step=0.01, format="%.2f")
monthly_charges = st.number_input("Biaya Bulanan (0.00–1.00)", min_value=0.00, max_value=1.00, value=0.00, step=0.01, format="%.2f")
total_charges = st.number_input("Total Biaya (0.00–1.00)", min_value=0.00, max_value=1.00, value=0.00, step=0.01, format="%.2f")

if st.button("Prediksi"):
    # Buat DataFrame input
    data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Prediksi langsung (karena data sudah dalam skala 0–1)
    pred = model.predict(data)
    proba = model.predict_proba(data)[0][1]

    st.subheader("Hasil Prediksi")
    if pred[0] == 1:
        st.error(f"⚠️ Pelanggan berpotensi CHURN (Probabilitas: {proba*100:.2f}%)")
    else:
        st.success(f"✅ Pelanggan diprediksi TIDAK CHURN (Probabilitas: {(1-proba)*100:.2f}%)")
