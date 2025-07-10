import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction App")

# Load Data
@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test

train, test = load_data()
test_ids = test['Id']

st.subheader("🔍 Initial Data Overview")
st.write("Train Shape:", train.shape)
st.write("Test Shape:", test.shape)

# Preprocessing
train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index, inplace=True)
y = np.log1p(train['SalePrice'])

train.drop(['SalePrice', 'Id'], axis=1, inplace=True)
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

# Fix Skewness
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.75].index
all_data[high_skew] = np.log1p(all_data[high_skew])

# One-hot encoding
all_data = pd.get_dummies(all_data)

# Split
X = all_data[:len(train)]
X_test = all_data[len(train):]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
ridge = Ridge(alpha=10)
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                   subsample=0.7, colsample_bytree=0.7, random_state=42)
lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                     subsample=0.7, colsample_bytree=0.7, random_state=42)

models = {"Ridge": ridge, "XGBoost": xgb, "LightGBM": lgbm}
rmse_scores = {}

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmse_scores[name] = rmse

# Stacked Model
estimators = [("ridge", ridge), ("xgb", xgb), ("lgbm", lgbm)]
stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stack.fit(X_train, y_train)
stack_preds = stack.predict(X_val)
stack_rmse = np.sqrt(mean_squared_error(y_val, stack_preds))
rmse_scores["Stacked"] = stack_rmse

# Select Champion
champion = min(rmse_scores, key=rmse_scores.get)
final_model = stack if champion == "Stacked" else models[champion]
final_model.fit(X_scaled, y)

st.subheader("📊 Model Performance (RMSE)")
for name, score in rmse_scores.items():
    st.write(f"{name}: {score:.4f}")
st.success(f"✅ Best Model: {champion} (RMSE: {rmse_scores[champion]:.4f})")

# Predictions
preds = np.expm1(final_model.predict(X_test_scaled))
submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})

st.subheader("📥 Download Prediction")
st.dataframe(submission.head())
st.download_button("📄 Download Submission CSV", submission.to_csv(index=False), file_name="submission.csv")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit by [Your Name]")
