import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")

# Load data
@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test

train, test = load_data()

# Preprocessing
train.drop(train[(train['GrLivArea'] > 4000) & (np.log1p(train['SalePrice']) < 12.6)].index, inplace=True)
y = np.log1p(train['SalePrice'])
train.drop(['SalePrice', 'Id'], axis=1, inplace=True)
test_ids = test['Id']
test.drop(['Id'], axis=1, inplace=True)
all_data = pd.concat([train, test], axis=0)

none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
             'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
             'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
for col in none_cols:
    all_data[col] = all_data[col].fillna("None")

zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
             'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

for col in all_data.columns:
    if all_data[col].dtype == 'object' and all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    elif all_data[col].isnull().sum() > 0:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# Ordinal Encoding
qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ord_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
            'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
for col in ord_cols:
    all_data[col] = all_data[col].map(qual_map)

# Feature Engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBathrooms'] = (all_data['FullBath'] + 0.5*all_data['HalfBath'] +
                               all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath'])

# Fix skew
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.75].index
all_data[high_skew] = np.log1p(all_data[high_skew])

# One-hot encoding
all_data = pd.get_dummies(all_data)

# Final split
X = all_data[:len(train)]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Models
ridge = Ridge(alpha=10)
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                   subsample=0.7, colsample_bytree=0.7, random_state=42)
lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                     subsample=0.7, colsample_bytree=0.7, random_state=42)

estimators = [("ridge", ridge), ("xgb", xgb), ("lgbm", lgbm)]
stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stack.fit(X_train, y_train)

# ============ User Input ============

st.subheader("🔧 Enter House Features to Predict Price")

# Let’s use only a few important numeric features from training data
input_features = {
    'OverallQual': st.slider("Overall Quality (1-10)", 1, 10, 5),
    'GrLivArea': st.slider("Above Grade Living Area (sq ft)", 400, 4000, 1500),
    'TotalBathrooms': st.slider("Total Bathrooms", 1.0, 5.0, 2.5),
    'GarageCars': st.slider("Garage Capacity (Cars)", 0, 4, 2),
    'GarageArea': st.slider("Garage Area (sq ft)", 0, 1000, 400),
    'TotalSF': st.slider("Total SF (Basement + 1st + 2nd)", 500, 6000, 1800),
    'YearBuilt': st.slider("Year Built", 1900, 2020, 2000),
    'YearRemodAdd': st.slider("Year Remodeled", 1950, 2020, 2005),
}

# Build input row with same structure as all_data
sample_input = pd.DataFrame([input_features])
# Fill in 0s for other features
model_features = pd.DataFrame(columns=X.columns)
input_final = pd.concat([sample_input, pd.DataFrame(columns=model_features.columns.difference(sample_input.columns))], axis=1).fillna(0)

# Scale
input_scaled = scaler.transform(input_final)

# Predict
if st.button("💡 Predict Price"):
    pred_log = stack.predict(input_scaled)[0]
    price = np.expm1(pred_log)
    st.success(f"🏠 Estimated House Price: **${price:,.0f}**")
