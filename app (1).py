import streamlit as st
import numpy as np
import joblib

st.title(" Bright Skies Predictor by Rawan Mohamed Farouk")
st.markdown("Estimate house sale price based on 5 features.")

# Load model and scaler safely
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(" Error loading model or scaler. Please make sure both 'model.pkl' and 'scaler.pkl' are in the app directory.")
    st.stop()

# Input form
st.sidebar.header("Enter Features")
OverallQual = st.sidebar.slider("Overall Quality (1–10)", 1, 10, 5)
GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", 400, 5000, 1500)
GarageCars = st.sidebar.slider("Garage Capacity", 0, 4, 2)
TotalBsmtSF = st.sidebar.number_input("Basement Area (sq ft)", 0, 3000, 800)
YearBuilt = st.sidebar.slider("Year Built", 1900, 2024, 2000)

if st.button(" Predict Sale Price"):
    input_data = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt]])
    input_scaled = scaler.transform(input_data)
    log_price = model.predict(input_scaled)[0]
    predicted_price = np.expm1(log_price)
    st.success(f" Estimated Sale Price: **${predicted_price:,.2f}**")
