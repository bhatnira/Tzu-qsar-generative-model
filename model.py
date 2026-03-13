"""
model.py
Module for model training, evaluation, and prediction.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import warnings
warnings.filterwarnings("ignore")

# Model factory with hyperparameters
model_factories = {
    "RF": lambda p: RandomForestRegressor(**p, random_state=42),
    "ET": lambda p: ExtraTreesRegressor(**p, random_state=42),
    "GB": lambda p: GradientBoostingRegressor(**p, random_state=42),
    "XGB": lambda p: xgb.XGBRegressor(**p, random_state=42, n_jobs=-1),
    "LGBM": lambda p: lgb.LGBMRegressor(**p, random_state=42),
    "CAT": lambda p: cat.CatBoostRegressor(**p, random_state=42, verbose=0, loss_function='RMSE'),
    "SVR": lambda p: SVR(**p),
    "KNN": lambda p: KNeighborsRegressor(**p),
    "MLP": lambda p: MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
    "Ridge": lambda p: Ridge(),
    "Lasso": lambda p: Lasso(),
    "ElasticNet": lambda p: ElasticNet(),
    "BayesianRidge": lambda p: BayesianRidge(),
    "Huber": lambda p: HuberRegressor(),
    "DT": lambda p: DecisionTreeRegressor(random_state=42)
}

# Models that require feature scaling
scaled_models = {"SVR","KNN","MLP","Ridge","Lasso","ElasticNet","BayesianRidge","Huber"}

def train_and_select(X_train_raw, y_train, X_test_raw, y_test):
    """
    Train multiple descriptor-model combinations and select the best performer.
    
    Args:
        X_train_raw: dict of descriptor matrices for training
        y_train: Target values (log-transformed IC50)
        X_test_raw: dict of descriptor matrices for testing
        y_test: Test target values
        
    Returns:
        tuple: (trained_models, imputers, scalers, best_desc, best_model_name, results_df)
    """
    imputers = {}
    scalers = {}
    trained_models = {}
    results = []
    
    for desc_name in X_train_raw:
        # Imputation
        imp = SimpleImputer(strategy="mean")
        X_train = imp.fit_transform(X_train_raw[desc_name])
        X_test = imp.transform(X_test_raw[desc_name])
        imputers[desc_name] = imp
        
        for model_name in model_factories:
            X_tr, X_te = X_train.copy(), X_test.copy()
            sc = None
            
            # Scale if needed
            if model_name in scaled_models:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
                scalers[(desc_name, model_name)] = sc
            
            # Train model
            model = model_factories[model_name]({})
            model.fit(X_tr, y_train)
            trained_models[(desc_name, model_name)] = model
            
            # Evaluate
            y_te_pred = model.predict(X_te)
            r2 = r2_score(y_test, y_te_pred)
            results.append({"Descriptor": desc_name, "Model": model_name, "R2": r2})
            
            print(f"Trained {model_name} with {desc_name}: R² = {r2:.4f}")
    
    results_df = pd.DataFrame(results)
    best_model_row = results_df.loc[results_df["R2"].idxmax()]
    best_desc = best_model_row["Descriptor"]
    best_model_name = best_model_row["Model"]
    
    print(f"\n✅ Best model: {best_model_name} using {best_desc} (R² = {best_model_row['R2']:.4f})")
    
    return trained_models, imputers, scalers, best_desc, best_model_name, results_df

def predict_and_antilog(model, X, df_original):
    """
    Make predictions and convert from log-space back to IC50 (nM).
    
    Args:
        model: Trained regression model
        X: Scaled descriptor matrix
        df_original: Original dataframe to attach predictions to
        
    Returns:
        DataFrame with added Predicted_IC50 column
    """
    y_pred_log = model.predict(X)
    y_pred = 10 ** y_pred_log  # Anti-log transformation
    df = df_original.copy()
    df["Predicted_IC50"] = y_pred
    return df

def save_predictions(predictions_dict, output_dir="outputs"):
    """
    Save predictions to CSV files.
    
    Args:
        predictions_dict: Dictionary with prediction dataframes
        output_dir: Output directory for CSV files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in predictions_dict.items():
        if not df.empty:
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(f"✅ Saved {filepath}")
