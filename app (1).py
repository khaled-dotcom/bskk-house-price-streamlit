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
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bright skies House Price Predictor by Rawan Mohamed", layout="centered")
st.title(" Bright skies House Price Prediction App by Rawan Mohamed")

@st.cache_data
def load_and_train():
    df = pd.read_csv("train.csv")

    df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index, inplace=True)
    y = np.log1p(df['SalePrice'])
    df.drop(['Id', 'SalePrice'], axis=1, inplace=True)

    # Fill missing
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Add engineered features
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                            df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])

    # Reduce skew
    numeric_feats = df.select_dtypes(include=[np.number]).columns
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75].index
    df[high_skew] = np.log1p(df[high_skew])

    # Dummies
    df = pd.get_dummies(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Models
    ridge = Ridge(alpha=10)
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    stack = StackingRegressor(estimators=[("ridge", ridge), ("xgb", xgb), ("lgbm", lgbm)],
                               final_estimator=LinearRegression())

    stack.fit(X_train, y_train)

    return df.columns.tolist(), scaler, stack

# Load model and scaler
feature_names, scaler, model = load_and_train()

st.header("🔧 Enter House Features")

#  Only 5 inputs
user_input = {
    'OverallQual': st.slider("Overall Quality (1-10)", 1, 10, 6),
    'GrLivArea': st.slider("Living Area (sq ft)", 500, 4000, 1500),
    'GarageCars': st.slider("Garage Capacity (Cars)", 0, 4, 2),
    'TotalSF': st.slider("Total Square Feet (Basement + Floors)", 500, 6000, 2000),
    'TotalBathrooms': st.slider("Total Bathrooms", 1.0, 5.0, 2.5)
}

# Build input DataFrame
input_df = pd.DataFrame([user_input])

# Fill other required columns with 0
full_input = pd.DataFrame(columns=feature_names)
for col in full_input.columns:
    full_input[col] = input_df[col] if col in input_df else 0

# Scale
scaled_input = scaler.transform(full_input)

# Predict
if st.button(" Predict House Price"):
    pred_log = model.predict(scaled_input)[0]
    price = np.expm1(pred_log)
    st.success(f" Estimated House Price: **${price:,.0f}**")
