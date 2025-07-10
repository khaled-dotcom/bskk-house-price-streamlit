import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bright skies  House Price Predictor by Rawan mohamed ", layout="centered")
st.title("Bright  skies House Price Prediction App by rawan mohamed ")

@st.cache_resource
def train_model_on_5_features():
    df = pd.read_csv("train.csv")

    # Clean outliers
    df = df[(df['GrLivArea'] < 4000) | (df['SalePrice'] > 300000)]

    # Select only 5 useful features
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']
    df = df[features + ['SalePrice']].dropna()

    df['TotalBathrooms'] = df['FullBath']  # Optional derived feature

    # Target: log price
    y = np.log1p(df['SalePrice'])
    X = df[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'TotalBathrooms']]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Models
    ridge = Ridge(alpha=10)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)

    stack = StackingRegressor(estimators=[
        ("ridge", ridge), ("xgb", xgb), ("lgbm", lgbm)
    ], final_estimator=LinearRegression())

    stack.fit(X_train, y_train)
    return scaler, stack

# Load once
scaler, model = train_model_on_5_features()

# --- User Input ---
st.header(" Enter House Features")

overall_qual = st.slider(" Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.slider(" Living Area (sq ft)", 500, 2500, 1500)
garage_cars = st.slider(" Garage Capacity", 0, 4, 2)
total_bsmtsf = st.slider(" Basement SF", 0, 2000, 800)
bathrooms = st.slider(" Bathrooms (full only)", 1, 4, 2)

input_data = pd.DataFrame([{
    'OverallQual': overall_qual,
    'GrLivArea': gr_liv_area,
    'GarageCars': garage_cars,
    'TotalBsmtSF': total_bsmtsf,
    'TotalBathrooms': bathrooms
}])

# Scale
scaled_input = scaler.transform(input_data)

# --- Prediction ---
if st.button(" Predict Price"):
    pred_log = model.predict(scaled_input)[0]

    if not np.isfinite(pred_log) or pred_log > 25:
        st.error(" Unstable prediction. Please adjust your input values.")
    else:
        price = np.expm1(pred_log)
        st.success(f" Estimated House Price: **${price:,.0f}**")
