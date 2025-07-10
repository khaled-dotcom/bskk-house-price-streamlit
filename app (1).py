import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="  Bright Skies", layout="centered")
st.title(" Bright Skies House Price Prediction App by Rawan mohamed")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("train.csv")

    # Clean outliers
    df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index, inplace=True)

    # Target
    y = np.log1p(df['SalePrice'])

    # Drop non-features
    df.drop(['Id', 'SalePrice'], axis=1, inplace=True)

    # Fill missing values
    none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
                'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
    for col in none_cols:
        df[col] = df[col].fillna("None")

    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
    for col in zero_cols:
        df[col] = df[col].fillna(0)

    for col in df.columns:
        if df[col].dtype == 'object' and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
        elif df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Map quality ratings
    qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    ord_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    for col in ord_cols:
        if col in df.columns:
            df[col] = df[col].map(qual_map)

    # Feature engineering
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = (df['FullBath'] + 0.5*df['HalfBath'] +
                            df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'])

    # Fix skew
    numeric_feats = df.select_dtypes(include=[np.number]).columns
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.75].index
    df[high_skew] = np.log1p(df[high_skew])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Models
    ridge = Ridge(alpha=10)
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)

    stack = StackingRegressor(estimators=[
        ("ridge", ridge), ("xgb", xgb), ("lgbm", lgbm)
    ], final_estimator=LinearRegression())

    stack.fit(X_train, y_train)

    return df.columns.tolist(), scaler, stack

# Load and train model once
feature_names, scaler, model = load_and_train()

# ========== UI ==========
st.header("🔧 Enter 5 Key Features")

user_input = {
    'OverallQual': st.slider("Overall Quality (1–10)", 1, 10, 6),
    'GrLivArea': st.slider("Living Area Above Ground (sq ft)", 400, 4000, 1500),
    'GarageCars': st.slider("Garage Capacity (Cars)", 0, 4, 2),
    'TotalSF': st.slider("Total Square Feet", 500, 6000, 2000),
    'TotalBathrooms': st.slider("Total Bathrooms", 1.0, 5.0, 2.5)
}

# Prepare user input row
input_df = pd.DataFrame([user_input])

# Build full input row with all required features
full_input = pd.DataFrame(columns=feature_names)

for col in full_input.columns:
    full_input[col] = input_df[col] if col in input_df.columns else 0

# Ensure no NaNs or wrong dtypes
full_input = full_input.fillna(0).astype(float)

# Scale
scaled_input = scaler.transform(full_input)

# Predict
if st.button(" Predict House Price"):
    try:
        pred_log = model.predict(scaled_input)[0]
        price = np.expm1(pred_log)
        st.success(f" Estimated House Price: **${price:,.0f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
