"""
XGBoost Modeling for ML Demand Prediction
=========================================

This script implements XGBoost modeling for demand prediction including:
- Feature engineering
- Hyperparameter tuning with Optuna
- Model training and evaluation
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
import optuna
import xgboost as xgb
import json
import os
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

def load_and_prepare_data(file_path="dataset/Synthetic_Demand_TimeSeries.csv"):
    """
    Load and prepare data for modeling.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (X, y, df_clean) - features, target, and cleaned dataframe
    """
    print("Loading data...")
    df = pd.read_csv(file_path, parse_dates=["Date"])
    
    # Remove initial lag NaN values
    df_clean = df.groupby("Location").apply(lambda x: x.iloc[3:]).reset_index(drop=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_clean.shape}")
    
    return df_clean

def perform_feature_engineering(df_clean):
    """
    Perform comprehensive feature engineering.
    
    Args:
        df_clean (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    print("Performing feature engineering...")
    
    # Calendar features
    df_clean["DayOfWeek"] = df_clean["Date"].dt.dayofweek
    df_clean["IsWeekend"] = df_clean["DayOfWeek"].isin([5, 6]).astype(int)
    
    # Holiday indicator
    us_holidays = pd.to_datetime([
        "2023-01-01", "2023-07-04", "2023-12-25",
        "2024-01-01", "2024-07-04", "2024-12-25",
    ])
    df_clean["IsHoliday"] = df_clean["Date"].isin(us_holidays).astype(int)
    
    # Additional lag features
    for lag in [7, 14, 30]:
        df_clean[f"Demand_lag_{lag}"] = (
            df_clean.groupby("Location")["Demand_Index"].shift(lag)
        )
    
    # Percent change + binning
    df_clean["Pct_change"] = (
        df_clean.groupby("Location")["Demand_Index"].pct_change()
    )
    
    df_clean["Pct_change_bin"] = pd.cut(
        df_clean["Pct_change"],
        bins=[-np.inf, -0.1, -0.02, 0.02, 0.1, np.inf],
        labels=["big_drop", "small_drop", "flat", "small_rise", "big_rise"]
    )
    
    pct_dummies = pd.get_dummies(df_clean["Pct_change_bin"], prefix="Pct_change_bin")
    df_clean = pd.concat([df_clean, pct_dummies], axis=1)
    df_clean = df_clean.drop(columns=["Pct_change", "Pct_change_bin"])
    
    # Region encoding
    df_clean["State"] = df_clean["Location"].str.split(",").str[1].str.strip()
    state_dummies = pd.get_dummies(df_clean["State"], prefix="State")
    df_clean = pd.concat([df_clean, state_dummies], axis=1)
    
    # Remove rows with NaN in lag features
    lag_cols = [col for col in df_clean.columns if "Demand_lag" in col or "Rolling_mean" in col]
    df_clean = df_clean.dropna(subset=lag_cols).reset_index(drop=True)
    
    print("✅ Feature engineering complete.")
    print(f"Shape after feature engineering: {df_clean.shape}")
    
    return df_clean

def prepare_features_and_target(df_clean):
    """
    Prepare features and target for modeling.
    
    Args:
        df_clean (pd.DataFrame): Dataframe with engineered features
        
    Returns:
        tuple: (X, y, features) - features, target, and feature names
    """
    exclude_cols = ["Date", "Location", "Demand_Index", "Forecasted_Demand", "State"]
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

def optimize_hyperparameters(X_train, y_train, X_test, y_test, n_trials=50):
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        X_train, y_train, X_test, y_test: Training and test data
        n_trials (int): Number of optimization trials
        
    Returns:
        dict: Best hyperparameters
    """
    print("Starting hyperparameter optimization...")
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train.values.astype(np.float64), label=y_train.values)
    dtest = xgb.DMatrix(X_test.values.astype(np.float64), label=y_test.values)
    
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "verbosity": 0,
            "seed": 42
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtest, "eval")],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        preds = model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial params:")
    print(study.best_trial.params)
    print(f"Best RMSE: {study.best_trial.value}")
    
    return study.best_trial.params

def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """
    Train the final model with best hyperparameters.
    
    Args:
        X_train, y_train, X_test, y_test: Training and test data
        best_params (dict): Best hyperparameters
        
    Returns:
        xgb.Booster: Trained model
    """
    print("Training final model...")
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train.values.astype(np.float64), label=y_train.values)
    dtest = xgb.DMatrix(X_test.values.astype(np.float64), label=y_test.values)
    
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtest, "eval")],
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    return final_model

def evaluate_model(model, X_test, y_test, X_encoded, train_size):
    """
    Evaluate the model and generate predictions.
    
    Args:
        model (xgb.Booster): Trained model
        X_test, y_test: Test data
        X_encoded (pd.DataFrame): Full encoded features
        train_size (int): Size of training set
        
    Returns:
        tuple: (y_pred_all, metrics) - predictions and evaluation metrics
    """
    # Predict on the entire dataset
    dall = xgb.DMatrix(X_encoded.values.astype(np.float64))
    y_pred_all = model.predict(dall)
    
    # Calculate metrics on test set only
    mse = mean_squared_error(y_test, y_pred_all[train_size:])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_all[train_size:])
    mape = mean_absolute_percentage_error(y_test, y_pred_all[train_size:]) * 100
    r2 = r2_score(y_test, y_pred_all[train_size:])
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }
    
    print("=======================================")
    print(f"Final RMSE: {rmse:.2f}")
    print(f"Final MAE: {mae:.2f}")
    print(f"Final MAPE: {mape:.2f}%")
    print(f"Final R2: {r2:.4f}")
    print("=======================================")
    
    return y_pred_all, metrics

def create_visualizations(y_test, y_pred_all, train_size, X_encoded, model):
    """
    Create model evaluation visualizations.
    
    Args:
        y_test: Test target values
        y_pred_all: All predictions
        train_size (int): Size of training set
        X_encoded (pd.DataFrame): Encoded features
        model (xgb.Booster): Trained model
    """
    print("Creating visualizations...")
    
    # 1. Predictions vs actuals
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual", alpha=0.7)
    plt.plot(y_pred_all[train_size:], label="Predicted", alpha=0.7)
    plt.legend()
    plt.title("Actual vs. Predicted Demand Index")
    plt.xlabel("Sample")
    plt.ylabel("Demand_Index")
    plt.show()
    
    # 2. Residual plot
    residuals = y_test - y_pred_all[train_size:]
    plt.figure(figsize=(8,5))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title("Residuals Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.show()
    
    # 3. Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_all[train_size:], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Demand_Index")
    plt.ylabel("Predicted Demand_Index")
    plt.title("Predicted vs. Actual Scatter Plot")
    plt.show()
    
    # 4. SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_encoded.values.astype(np.float64))
    
    shap.summary_plot(shap_values, X_encoded, max_display=15, show=False)
    plt.title("SHAP Summary Plot - XGBoost")
    plt.show()
    
    # 5. Top feature dependence plot
    top_feature_idx = np.abs(shap_values).mean(axis=0).argmax()
    top_feature = X_encoded.columns[top_feature_idx]
    shap.dependence_plot(top_feature, shap_values, X_encoded)
    
    print("✅ Visualizations created!")

def save_model_and_outputs(model, X_train, df_clean, y_pred_all, features):
    """
    Save model and generate outputs.
    
    Args:
        model (xgb.Booster): Trained model
        X_train (pd.DataFrame): Training features
        df_clean (pd.DataFrame): Cleaned dataframe
        y_pred_all (np.array): All predictions
        features (list): Feature names
    """
    print("Saving model and outputs...")
    
    # Create output directory if it doesn't exist
    os.makedirs("Output/ML_Forecasting_Results", exist_ok=True)
    
    # Save model
    model.save_model("Output/ML_Forecasting_Results/xgb_model.json")
    print("✅ Model saved: Output/ML_Forecasting_Results/xgb_model.json")
    
    # Save feature names
    feature_names = list(X_train.columns)
    with open("Output/ML_Forecasting_Results/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print("✅ Feature names saved: Output/ML_Forecasting_Results/feature_names.json")
    
    # Save predictions to dataframe
    df_clean["Forecasted_Demand"] = y_pred_all
    
    # Regional aggregation
    regional_df = (
        df_clean.groupby("Location")["Forecasted_Demand"]
        .mean()
        .reset_index()
        .rename(columns={"Forecasted_Demand": "Forecasted_Demand_Index"})
    )
    
    regional_df.to_csv("Output/ML_Forecasting_Results/Regional_Demand_Index.csv", index=False)
    print("✅ Regional demand index saved: Output/ML_Forecasting_Results/Regional_Demand_Index.csv")
    
    # Weighted averages by location
    final_df_sorted = df_clean.sort_values(["Location", "Date"])
    final_df_sorted["Weight"] = final_df_sorted.groupby("Location").cumcount() + 1
    
    location_weighted_avg = (
        final_df_sorted
        .groupby("Location")
        .apply(lambda g: pd.Series({
            "Weighted_Avg_Demand_Index":
                np.average(g["Demand_Index"], weights=g["Weight"]),
            "Weighted_Avg_Forecasted_Demand":
                np.average(g["Forecasted_Demand"], weights=g["Weight"])
        }))
        .reset_index()
    )
    
    location_weighted_avg.to_csv("Output/ML_Forecasting_Results/Weighted_Avg_Demand_Index_by_Location.csv", index=False)
    print("✅ Weighted averages saved: Output/ML_Forecasting_Results/Weighted_Avg_Demand_Index_by_Location.csv")
    
    print(f"Number of regions: {regional_df.shape[0]}")
    print("\nSample regional results:")
    print(regional_df.head())

def run_complete_modeling_pipeline():
    """
    Run the complete XGBoost modeling pipeline.
    """
    print("="*60)
    print("XGBOOST MODELING PIPELINE")
    print("="*60)
    
    # Step 1: Load and prepare data
    df_clean = load_and_prepare_data()
    
    # Step 2: Feature engineering
    df_clean = perform_feature_engineering(df_clean)
    
    # Step 3: Prepare features and target
    X, y, features = prepare_features_and_target(df_clean)
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test, train_size = split_data(X, y)
    
    # Step 5: Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test)
    
    # Step 6: Train final model
    final_model = train_final_model(X_train, y_train, X_test, y_test, best_params)
    
    # Step 7: Evaluate model
    y_pred_all, metrics = evaluate_model(final_model, X_test, y_test, X, train_size)
    
    # Step 8: Create visualizations
    create_visualizations(y_test, y_pred_all, train_size, X, final_model)
    
    # Step 9: Save model and outputs
    save_model_and_outputs(final_model, X_train, df_clean, y_pred_all, features)
    
    print("\n" + "="*60)
    print("MODELING PIPELINE COMPLETED!")
    print("="*60)
    
    return final_model, metrics, df_clean

if __name__ == "__main__":
    # Run complete modeling pipeline
    model, metrics, df_clean = run_complete_modeling_pipeline() 