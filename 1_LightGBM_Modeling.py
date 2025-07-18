"""
LightGBM Modeling for ML Demand Prediction
==========================================

This script implements LightGBM modeling for demand prediction including:
- Data loading and preprocessing
- LightGBM model training
- Model evaluation and visualization
- SHAP analysis
- Model saving and outputs

Author: Leanna Jeon
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb
import os
import json
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

def load_and_prepare_data(file_path="dataset/Synthetic_Demand_TimeSeries.csv"):
    """
    Load and prepare data for LightGBM modeling.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (X, y, df_clean) - features, target, and cleaned dataframe
    """
    print("Loading data for LightGBM modeling...")
    df = pd.read_csv(file_path, parse_dates=["Date"])
    
    # Remove initial lag NaN values
    df_clean = df.groupby("Location").apply(lambda x: x.iloc[3:]).reset_index(drop=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_clean.shape}")
    
    return df_clean

def prepare_features_and_target(df_clean):
    """
    Prepare features and target for LightGBM modeling.
    
    Args:
        df_clean (pd.DataFrame): Cleaned dataframe
        
    Returns:
        tuple: (X, y, features) - features, target, and feature names
    """
    exclude_cols = [
        "Date", "Location", "Demand_Index", "Forecasted_Demand"
    ]
    
    features = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[features]
    y = df_clean["Demand_Index"]
    
    print(f"Number of features: {len(features)}")
    print(f"Target variable: Demand_Index")
    
    return X, y, features

def split_data(X, y, train_ratio=0.8):
    """
    Split data into train and test sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        train_ratio (float): Ratio for training set
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_size)
    """
    train_size = int(len(X) * train_ratio)
    
    X_train = X.iloc[:train_size, :]
    y_train = y.iloc[:train_size]
    
    X_test = X.iloc[train_size:, :]
    y_test = y.iloc[train_size:]
    
    print(f"Train rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, train_size

def train_lightgbm_model(X_train, y_train, X_test, y_test, params=None):
    """
    Train LightGBM model with given parameters.
    
    Args:
        X_train, y_train, X_test, y_test: Training and test data
        params (dict): LightGBM parameters
        
    Returns:
        lgb.LGBMRegressor: Trained LightGBM model
    """
    print("Training LightGBM model...")
    
    if params is None:
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
    
    lgb_model = lgb.LGBMRegressor(**params)
    
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )
    
    print("✅ LightGBM model training completed!")
    return lgb_model

def evaluate_lightgbm_model(model, X_test, y_test, df_clean, train_size):
    """
    Evaluate LightGBM model and generate predictions.
    
    Args:
        model (lgb.LGBMRegressor): Trained LightGBM model
        X_test, y_test: Test data
        df_clean (pd.DataFrame): Cleaned dataframe
        train_size (int): Size of training set
        
    Returns:
        tuple: (y_pred_lgb, metrics) - predictions and evaluation metrics
    """
    # Generate predictions
    y_pred_lgb = model.predict(X_test)
    
    # Save predictions to dataframe
    df_clean.loc[df_clean.index[train_size:], "Forecasted_Demand_LGB"] = y_pred_lgb
    
    # Calculate metrics
    mse_lgb = mean_squared_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mse_lgb)
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    mape_lgb = mean_absolute_percentage_error(y_test, y_pred_lgb) * 100
    r2_lgb = r2_score(y_test, y_pred_lgb)
    
    metrics = {
        'RMSE': rmse_lgb,
        'MAE': mae_lgb,
        'MAPE': mape_lgb,
        'R2': r2_lgb
    }
    
    print("=======================================")
    print(f"✅ LightGBM Test RMSE: {rmse_lgb:.2f}")
    print(f"✅ LightGBM Test MAE: {mae_lgb:.2f}")
    print(f"✅ LightGBM Test MAPE: {mape_lgb:.2f}%")
    print(f"✅ LightGBM Test R2: {r2_lgb:.4f}")
    print("=======================================")
    
    return y_pred_lgb, metrics

def create_lightgbm_visualizations(y_test, y_pred_lgb, X_test, model):
    """
    Create LightGBM model evaluation visualizations.
    
    Args:
        y_test: Test target values
        y_pred_lgb: LightGBM predictions
        X_test: Test features
        model (lgb.LGBMRegressor): Trained LightGBM model
    """
    print("Creating LightGBM visualizations...")
    
    # 1. Predictions vs actuals
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual", alpha=0.7)
    plt.plot(y_pred_lgb, label="Predicted (LightGBM)", alpha=0.7)
    plt.legend()
    plt.title("Actual vs. Predicted Demand Index - LightGBM")
    plt.xlabel("Sample")
    plt.ylabel("Demand_Index")
    plt.show()
    
    # 2. Residual plot
    residuals = y_test - y_pred_lgb
    plt.figure(figsize=(8,5))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title("Residuals Distribution - LightGBM")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.show()
    
    # 3. Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_lgb, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Demand_Index")
    plt.ylabel("Predicted Demand_Index")
    plt.title("Predicted vs. Actual Scatter Plot - LightGBM")
    plt.show()
    
    # 4. SHAP Analysis
    booster = model.booster_
    explainer_lgb = shap.TreeExplainer(booster)
    shap_values_lgb = explainer_lgb.shap_values(X_test)
    
    shap.summary_plot(shap_values_lgb, X_test, max_display=15, show=False)
    plt.title("SHAP Summary Plot - LightGBM", fontsize=16)
    plt.show()
    
    # 5. Top feature dependence plot
    top_feature = X_test.columns[np.abs(shap_values_lgb).mean(0).argmax()]
    shap.dependence_plot(top_feature, shap_values_lgb, X_test, show=False)
    plt.title(f"SHAP Dependence Plot - {top_feature}", fontsize=16)
    plt.show()
    
    print("✅ LightGBM visualizations created!")

def save_lightgbm_model_and_outputs(model, X_train, df_clean, y_pred_lgb, features):
    """
    Save LightGBM model and generate outputs.
    
    Args:
        model (lgb.LGBMRegressor): Trained LightGBM model
        X_train (pd.DataFrame): Training features
        df_clean (pd.DataFrame): Cleaned dataframe
        y_pred_lgb (np.array): LightGBM predictions
        features (list): Feature names
    """
    print("Saving LightGBM model and outputs...")
    
    # Create output directory if it doesn't exist
    os.makedirs("ML_outputs", exist_ok=True)
    
    # Save model
    model.save_model("ML_outputs/lightgbm_model.txt")
    print("✅ LightGBM model saved: ML_outputs/lightgbm_model.txt")
    
    # Save feature names
    feature_names = list(X_train.columns)
    with open("ML_outputs/lightgbm_feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print("✅ LightGBM feature names saved: ML_outputs/lightgbm_feature_names.json")
    
    # Regional aggregation for LightGBM
    regional_df = (
        df_clean.groupby("Location")["Forecasted_Demand_LGB"]
        .mean()
        .reset_index()
        .rename(columns={"Forecasted_Demand_LGB": "Forecasted_Demand_Index_LGB"})
    )
    
    regional_df.to_csv("ML_outputs/Regional_Demand_Index_LightGBM.csv", index=False)
    print("✅ LightGBM regional demand index saved: ML_outputs/Regional_Demand_Index_LightGBM.csv")
    
    # Weighted averages by location for LightGBM
    final_df_sorted = df_clean.sort_values(["Location", "Date"])
    final_df_sorted["Weight"] = final_df_sorted.groupby("Location").cumcount() + 1
    
    location_weighted_avg = (
        final_df_sorted
        .groupby("Location")
        .apply(lambda g: pd.Series({
            "Weighted_Avg_Demand_Index":
                np.average(g["Demand_Index"], weights=g["Weight"]),
            "Weighted_Avg_Forecasted_Demand_LGB":
                np.average(g["Forecasted_Demand_LGB"], weights=g["Weight"])
        }))
        .reset_index()
    )
    
    location_weighted_avg.to_csv("ML_outputs/Weighted_Avg_Demand_Index_LightGBM.csv", index=False)
    print("✅ LightGBM weighted averages saved: ML_outputs/Weighted_Avg_Demand_Index_LightGBM.csv")
    
    print(f"Number of regions: {regional_df.shape[0]}")
    print("\nSample LightGBM regional results:")
    print(regional_df.head())

def run_lightgbm_pipeline():
    """
    Run the complete LightGBM modeling pipeline.
    """
    print("="*60)
    print("LIGHTGBM MODELING PIPELINE")
    print("="*60)
    
    # Step 1: Load and prepare data
    df_clean = load_and_prepare_data()
    
    # Step 2: Prepare features and target
    X, y, features = prepare_features_and_target(df_clean)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test, train_size = split_data(X, y)
    
    # Step 4: Train LightGBM model
    lgb_model = train_lightgbm_model(X_train, y_train, X_test, y_test)
    
    # Step 5: Evaluate model
    y_pred_lgb, metrics = evaluate_lightgbm_model(lgb_model, X_test, y_test, df_clean, train_size)
    
    # Step 6: Create visualizations
    create_lightgbm_visualizations(y_test, y_pred_lgb, X_test, lgb_model)
    
    # Step 7: Save model and outputs
    save_lightgbm_model_and_outputs(lgb_model, X_train, df_clean, y_pred_lgb, features)
    
    print("\n" + "="*60)
    print("LIGHTGBM MODELING PIPELINE COMPLETED!")
    print("="*60)
    
    return lgb_model, metrics, df_clean

if __name__ == "__main__":
    # Run complete LightGBM modeling pipeline
    model, metrics, df_clean = run_lightgbm_pipeline() 