import streamlit as st
import numpy as np
import pickle

# Load model and scaler using pickle
with open("stack_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title(" Bright Skies Predictor by Rawan Mohamed Farouk")

st.sidebar.header("Input Features")
OverallQual = st.sidebar.slider("Overall Quality", 1, 10, 5)
GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", 400, 5000, 1500)
GarageCars = st.sidebar.slider("Garage Capacity", 0, 4, 2)
TotalBsmtSF = st.sidebar.number_input("Total Basement Area", 0, 3000, 800)
YearBuilt = st.sidebar.slider("Year Built", 1900, 2024, 2000)

if st.button("🔍 Predict Sale Price"):
    features = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt]])
    scaled = scaler.transform(features)
    log_price = model.predict(scaled)[0]
    price = np.expm1(log_price)
    st.success(f" Estimated Sale Price: **${price:,.2f}**")
