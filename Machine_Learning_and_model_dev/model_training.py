
# model_training.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump, load

# --- Setup ---
# Ensure models folder exists
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


# --- Helper Functions ---

def create_pipeline(model, model_type='classification'):
    """Create preprocessing + model pipeline."""
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    steps = [('imputer', imputer), ('scaler', scaler)]
    if model_type == 'classification':
        steps.append(('classifier', model))
    else:
        steps.append(('regressor', model))
    return Pipeline(steps)


def get_model_path(model_choice):
    """Generate safe model path based on model name."""
    safe_model_name = model_choice.lower().replace(' ', '_')
    model_path = os.path.join(MODELS_DIR, f'{safe_model_name}.joblib')
    return model_path


def train_and_save_model(model_choice, params, X_train, y_train):
    """Train and save the selected model."""
    if model_choice == 'Random Forest Classifier':
        model = RandomForestClassifier(random_state=42, **params)
        pipeline = create_pipeline(model, model_type='classification')
    elif model_choice == 'Gradient Boosting Regression':
        model = GradientBoostingRegressor(random_state=42, **params)
        pipeline = create_pipeline(model, model_type='regression')
    elif model_choice == 'Ridge Regression':
        model = Ridge(**params)
        pipeline = create_pipeline(model, model_type='regression')
    elif model_choice == 'Lasso Regression':
        model = Lasso(**params)
        pipeline = create_pipeline(model, model_type='regression')
    elif model_choice == 'Support Vector Machine':
        model = SVC(random_state=42, gamma=0.5, **params)
        pipeline = create_pipeline(model, model_type='classification')
    else:
        raise ValueError(f"Unsupported model: {model_choice}")

    # Fit and save
    pipeline.fit(X_train, y_train)
    model_path = get_model_path(model_choice)
    dump(pipeline, model_path)
    return pipeline


def load_model(model_choice):
    """Load a saved model based on model choice."""
    model_path = get_model_path(model_choice)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")
    return load(model_path)


def predict(model, X_test):
    """Make predictions."""
    return model.predict(X_test)


def calculate_metrics(model_choice, y_true, y_pred, y_class=None):
    """Calculate evaluation metrics."""
    metrics = {}
    if model_choice in ['Ridge Regression', 'Lasso Regression', 'Gradient Boosting Regression']:
        # Regression metrics
        y_pred_exp = np.expm1(y_pred)  # Reverse log1p if needed
        metrics['R2'] = r2_score(y_true, y_pred_exp)
        metrics['MAE'] = mean_absolute_error(y_true, y_pred_exp)
        mse = mean_squared_error(y_true, y_pred_exp)
        metrics['RMSE'] = np.sqrt(mse)
    else:
        # Classification metrics
        if y_class is not None:
            metrics['Accuracy'] = accuracy_score(y_class, y_pred)
            metrics['F1'] = f1_score(y_class, y_pred)
    return metrics


def list_saved_models():
    """List all saved models nicely."""
    if not os.path.exists(MODELS_DIR):
        return []
    
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')]
    model_names = [f.replace('.joblib', '').replace('_', ' ').title() for f in model_files]
    return model_names


def list_saved_models_full():
    """List saved models with full paths."""
    if not os.path.exists(MODELS_DIR):
        return []
    
    model_list = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.joblib'):
            nice_name = f.replace('.joblib', '').replace('_', ' ').title()
            full_path = os.path.join(MODELS_DIR, f)
            model_list.append({'name': nice_name, 'path': full_path})
    return model_list