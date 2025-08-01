import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold

df = pd.read_csv("../Weather_&_Climate_Data/Processed Datas/Feature_Engineering/climate_df_scaled.csv")

# Calculate temperature anomaly (target for regression)
historical_avg_temp = df['Temp_2m_scaled'].mean()
df['Temp_Anomaly'] = df['Temp_2m_scaled'] - historical_avg_temp

X = df[['Humidity_2m_scaled', 'Pressure_scaled', 'WindSpeed_10m_scaled']]  # Features
y = df['Temp_Anomaly']                                # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)
    
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Ridge Regression (L2 penalty)
model_ridge = Ridge(alpha=1.0)  # Alpha controls regularization strength
model_ridge.fit(X_train, y_train)

# Lasso Regression (L1 penalty, for feature selection)
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X_train, y_train)

# Compare coefficients
print("Ridge Coefficients:", model_ridge.coef_)
print("Lasso Coefficients:", model_lasso.coef_)  # Sparse output

model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)

print(f"XGBoost RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")

# 5-fold cross-validation for XGBoost
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model_xgb, X, y, cv=kfold, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"CV RMSE: {rmse_scores.mean():.2f} (±{rmse_scores.std():.2f})")