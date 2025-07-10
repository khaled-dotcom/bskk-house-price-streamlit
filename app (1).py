import os
import importlib.util
import streamlit as st
import numpy as np

# Force install joblib if missing
os.system("pip install --user joblib")

# Dynamically load joblib after install
spec = importlib.util.find_spec("joblib")
if spec is None:
    st.error(" Failed to install joblib. Try restarting the app or check requirements.")
    st.stop()
else:
    joblib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(joblib)

# App title
st.title("🏡 Bright Skies Predictor by Rawan Mohamed Farouk")
st.markdown("Estimate house sale price based on 5 features.")

# Load model and scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(" Could not load model or scaler. Make sure 'model.pkl' and 'scaler.pkl' exist.")
    st.exception(e)
    st.stop()

# Inputs
st.sidebar.header("Input Features")
OverallQual = st.sidebar.slider("Overall Quality (1–10)", 1, 10, 5)
GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", 400, 5000, 1500)
GarageCars = st.sidebar.slider("Garage Capacity", 0, 4, 2)
TotalBsmtSF = st.sidebar.number_input("Basement Area (sq ft)", 0, 3000, 800)
YearBuilt = st.sidebar.slider("Year Built", 1900, 2024, 2000)

# Predict button
if st.button("🔍 Predict Sale Price"):
    try:
        input_data = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt]])
        input_scaled = scaler.transform(input_data)
        log_price = model.predict(input_scaled)[0]
        predicted_price = np.expm1(log_price)
        st.success(f" Estimated Sale Price: **${predicted_price:,.2f}**")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
