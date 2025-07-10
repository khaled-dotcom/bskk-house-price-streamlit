
import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Bright skies predictor by Rawan Mohamed Farouk")

# Sidebar inputs
st.sidebar.header("Input Features")

OverallQual = st.sidebar.slider("Overall Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)
GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=400, max_value=5000, value=1500)
GarageCars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
TotalBsmtSF = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
YearBuilt = st.sidebar.slider("Year Built", 1900, 2024, 2000)

# Make prediction
if st.button("Predict Sale Price"):
    features = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt]])
    scaled_features = scaler.transform(features)
    log_price = model.predict(scaled_features)[0]
    predicted_price = np.expm1(log_price)

    st.success(f" Estimated Sale Price: **${predicted_price:,.2f}**")
