import streamlit as st
import numpy as np

# Try importing joblib safely
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # for older sklearn versions

# Title
st.title(" Bright Skies Predictor by Rawan Mohamed Farouk")

st.markdown("Predict the sale price of a house based on 5 key features.")

# Try loading model and scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(" Could not load model or scaler. Please ensure both 'model.pkl' and 'scaler.pkl' exist in the same folder.")
    st.stop()

# Sidebar for input
st.sidebar.header("Input House Features")

OverallQual = st.sidebar.slider("Overall Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)
GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=400, max_value=5000, value=1500)
GarageCars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
TotalBsmtSF = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
YearBuilt = st.sidebar.slider("Year Built", 1900, 2024, 2000)

# Predict button
if st.button(" Predict Sale Price"):
    try:
        input_data = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt]])
        input_scaled = scaler.transform(input_data)
        log_price = model.predict(input_scaled)[0]
        price = np.expm1(log_price)
        st.success(f" Estimated Sale Price: **${price:,.2f}**")
    except Exception as e:
        st.error(" Prediction failed. Check the model or input formatting.")
        st.exception(e)
