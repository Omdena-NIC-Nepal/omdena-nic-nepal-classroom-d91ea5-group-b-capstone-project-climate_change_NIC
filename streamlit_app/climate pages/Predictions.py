
# 3_Predictions.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add the src directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to project root, then into Machine_Learning_and_Model_Training
project_root = Path(current_dir).parent
ml_dir = os.path.join(project_root, 'Machine_Learning_and_model_dev')

# Add to Python path
sys.path.append(str(ml_dir))

#import functions
from model_training import train_and_save_model, predict, calculate_metrics, load_model
# --- Helper Functions ---

@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Interval_Start'])
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])

        # Feature Engineering
        features = ['year', 'month', 'day', 'DayOfYear', 'WeekOfYear', 'Sin_Month', 'Cos_Month']
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Sin_Month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['Cos_Month'] = np.cos(2 * np.pi * df['month'] / 12)

        y = df['Temp_2m']

        df_clean = df.dropna(subset=features + ['Temp_2m'])
        X_clean = df_clean[features]
        y_clean = df_clean['Temp_2m']

        y_class = ((y_clean > 28) | (y_clean < 0)).astype(int)
        y_clean = y_clean.where(y_clean > 0, 1e-6)
        y_log = np.log1p(y_clean)
        return df_clean, X_clean, y_clean, y_log, y_class
    except Exception as e:
        st.error(f"❗ Error loading data: {e}")
        return None, None, None, None, None

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled

# Bonus Pro Trick: Cache model training (only retrain if inputs change!)
@st.cache_resource
def cached_train_model(model_choice, params, X_scaled, y_log):
    return train_and_save_model(model_choice, params, X_scaled, y_log)

# --- Streamlit App ---

st.title("🌎 Climate Prediction and Assessment App")

# --- Load Data Dynamically ---
current_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parent
csv_path = os.path.join(current_dir, 'Weather_&_Climate_Data', 'Processed Datas', '5yeargrouped.csv')

if not os.path.exists(csv_path):
    st.error(f"❗ Data file not found at expected location: `{csv_path}`")
    st.stop()

df, X, y, y_log, y_class = load_data(csv_path)
if df is None:
    st.stop()

# --- Model selection ---
model_choice = st.selectbox("Choose Model", [
    'Ridge Regression',
    'Lasso Regression',
    'Gradient Boosting Regression',
    'Random Forest Classifier',
    'Support Vector Machine'
])

# --- Hyperparameters ---
params = {}
run_fast = st.checkbox("⚡ Run Fast Mode", value=True)

if model_choice in ["Ridge Regression", "Lasso Regression"]:
    params['alpha'] = st.slider("Alpha", 0.01, 10.0, 1.0)
elif model_choice == "Gradient Boosting Regression":
    params['n_estimators'] = st.slider("Estimators", 50, 1000, 100)
    params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1)
    params['max_depth'] = st.slider("Max Depth", 2, 10, 3)
elif model_choice == "Random Forest Classifier":
    params['n_estimators'] = st.slider("Estimators", 50, 500, 100)
elif model_choice == "Support Vector Machine":
    params['C'] = st.slider("Penalty (C)", 0.01, 10.0, 1.0)

if run_fast:
    if model_choice in ["Ridge Regression", "Lasso Regression"]:
        params['alpha'] = 0.1
    elif model_choice == "Gradient Boosting Regression":
        params['n_estimators'] = 50
        params['learning_rate'] = 0.2
        params['max_depth'] = 3
    elif model_choice == "Random Forest Classifier":
        params['n_estimators'] = 100
    elif model_choice == "Support Vector Machine":
        params['C'] = 0.1

# --- Train and Predict ---
model_trained = None
scaler = None

if st.button("🚀 Train and Predict"):
    with st.spinner("Training model..."):
        scaler, X_scaled = scale_features(X)

        # Choose correct target
        if model_choice in ["Random Forest Classifier", "Support Vector Machine"]:
            y_target = y_class
        else:
            y_target = y_log

        model_trained = cached_train_model(model_choice, params, X_scaled, y_target)

        # Save model to session state
        st.session_state['model_trained'] = model_trained
        st.session_state['scaler'] = scaler

        # Predict on training data
        y_pred = predict(model_trained, X_scaled)
        metrics = calculate_metrics(model_choice, y, y_pred, y_class)

        if model_choice in ["Ridge Regression", "Lasso Regression", "Gradient Boosting Regression"]:
            st.success(f"🔵 R²: {metrics['R2']:.3f} | 🔵 MAE: {metrics['MAE']:.2f} | 🔵 RMSE: {metrics['RMSE']:.2f}")
        else:
            st.success(f"🟢 F1 Score: {metrics['F1']:.3f} | 🟢 Accuracy: {metrics['Accuracy']:.3f}")

# --- Future Prediction ---
st.subheader("📅 Set Future Prediction Date")

future_year = st.slider("Predict Year", 2025, 2035, 2026)
future_month = st.selectbox("Predict Month", range(1, 13))
future_day = st.slider("Predict Day", 1, 28, 15)

if st.button("📅 Make Future Predictions"):
    if 'model_trained' not in st.session_state:
        st.error("❗ Please train a model first.")
        st.stop()

    model_trained = st.session_state['model_trained']
    scaler = st.session_state['scaler']

    # Prepare input for prediction
    future_input = pd.DataFrame({'Date': [pd.Timestamp(future_year, future_month, future_day)]})
    features = ['year', 'month', 'day', 'DayOfYear', 'WeekOfYear', 'Sin_Month', 'Cos_Month']

    future_input['year'] = future_input['Date'].dt.year
    future_input['month'] = future_input['Date'].dt.month
    future_input['day'] = future_input['Date'].dt.day
    future_input['DayOfYear'] = future_input['Date'].dt.dayofyear
    future_input['WeekOfYear'] = future_input['Date'].dt.isocalendar().week
    future_input['Sin_Month'] = np.sin(2 * np.pi * future_input['month'] / 12)
    future_input['Cos_Month'] = np.cos(2 * np.pi * future_input['month'] / 12)

    # Scale using existing scaler
    future_scaled = scaler.transform(future_input[features])

    future_pred = predict(model_trained, future_scaled)

    if model_choice in ["Ridge Regression", "Lasso Regression", "Gradient Boosting Regression"]:
        future_temp = np.expm1(future_pred)
        st.success(f"📅 Predicted Temperature for {future_day}/{future_month}/{future_year}: **{future_temp[0]:.2f}°C**")
    else:
        label = 'Extreme Event' if future_pred[0] == 1 else 'Normal'
        st.success(f"📅 Predicted Climate Event for {future_day}/{future_month}/{future_year}: **{label}**")