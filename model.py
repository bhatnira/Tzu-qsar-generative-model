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

model_factories = {
    "RF": lambda p: RandomForestRegressor(**p, random_state=42),
    "ET": lambda p: ExtraTreesRegressor(**p, random_state=42),
    "GB": lambda p: GradientBoostingRegressor(**p, random_state=42),
    "XGB": lambda p: xgb.XGBRegressor(**p, random_state=42, n_jobs=-1),
    "LGBM": lambda p: lgb.LGBMRegressor(**p, random_state=42),
    "CAT": lambda p: cat.CatBoostRegressor(**p, random_state=42, verbose=0, loss_function='RMSE'),
    "SVR": lambda p: SVR(**p),
    "KNN": lambda p: KNeighborsRegressor(**p),
    "MLP": lambda p: MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500),
    "Ridge": lambda p: Ridge(),
    "Lasso": lambda p: Lasso(),
    "ElasticNet": lambda p: ElasticNet(),
    "BayesianRidge": lambda p: BayesianRidge(),
    "Huber": lambda p: HuberRegressor(),
    "DT": lambda p: DecisionTreeRegressor(random_state=42)
}

scaled_models = {"SVR","KNN","MLP","Ridge","Lasso","ElasticNet","BayesianRidge","Huber"}

def train_and_select(X_train_raw, y_train, X_test_raw, y_test):
    imputers = {}
    scalers = {}
    trained_models = {}
    results = []
    for desc_name in X_train_raw:
        imp = SimpleImputer(strategy="mean")
        X_train = imp.fit_transform(X_train_raw[desc_name])
        X_test = imp.transform(X_test_raw[desc_name])
        imputers[desc_name] = imp
        for model_name in model_factories:
            X_tr, X_te = X_train.copy(), X_test.copy()
            sc = None
            if model_name in scaled_models:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
                scalers[(desc_name, model_name)] = sc
            model = model_factories[model_name]({})
            model.fit(X_tr, y_train)
            trained_models[(desc_name, model_name)] = model
            y_te_pred = model.predict(X_te)
            r2 = r2_score(y_test, y_te_pred)
            results.append({"Descriptor": desc_name, "Model": model_name, "R2": r2})
    results_df = pd.DataFrame(results)
    best_model_row = results_df.loc[results_df["R2"].idxmax()]
    best_desc = best_model_row["Descriptor"]
    best_model_name = best_model_row["Model"]
    return trained_models, imputers, scalers, best_desc, best_model_name, results_df

def predict_and_antilog(model, X, df_original):
    y_pred_log = model.predict(X)
    y_pred = 10 ** y_pred_log
    df = df_original.copy()
    df["Predicted_IC50"] = y_pred
    return df
