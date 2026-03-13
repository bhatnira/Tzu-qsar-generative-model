"""
build_qsar_artifact.py
Train a QSAR model on Series A/B/D data, evaluate on Series C,
and save the best model as reinvent_integration/artifacts/qsar_best_model.joblib
in the format expected by reinvent_qsar_pharm_score.py:
  {
      "descriptor_name": str,
      "imputer":         sklearn SimpleImputer,
      "scaler":          sklearn StandardScaler | None,
      "model":           fitted sklearn/xgb/lgbm model,
  }
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from data_loader import load_excel_sheets, apply_smiles_cleaning, combine_and_deduplicate, filter_numeric_ic50
from descriptors import compute_descriptors
from model import model_factories, scaled_models

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
EXCEL = ROOT / "TB Project QSAR.xlsx"
if not EXCEL.exists():
    raise FileNotFoundError(f"Cannot find {EXCEL}")

print("[1/5] Loading and cleaning data...")
series_dfs = load_excel_sheets(str(EXCEL))
series_dfs = apply_smiles_cleaning(series_dfs)
df = combine_and_deduplicate(series_dfs)
df = filter_numeric_ic50(df)

# The scorer computes pIC50 = 6 - model_output, so the model must predict
# log10(IC50_uM) (same as main.py's transformed_IC50 = log10(IC50_uM)).
df["pIC50"] = np.log10(df["IC50 uM"].astype(float))

# Map series codes
code_col = "Series_Code" if "Series_Code" in df.columns else "Series"
series_codes = df[code_col].str.strip().str.upper()

train_df   = df[series_codes.isin(["A", "B", "D"])].reset_index(drop=True)
test_df    = df[series_codes == "C"].reset_index(drop=True)

if len(train_df) == 0:
    raise RuntimeError("No training data found for Series A/B/D")
if len(test_df) == 0:
    raise RuntimeError("No test data found for Series C")

print(f"    Train: {len(train_df)} compounds (A/B/D)  |  Test: {len(test_df)} compounds (C)")

smiles_col = "Canonical_SMILES" if "Canonical_SMILES" in df.columns else "Canonical SMILES"
smiles_train = train_df[smiles_col].tolist()
smiles_test  = test_df[smiles_col].tolist()
y_train = train_df["pIC50"].values.astype(float)
y_test  = test_df["pIC50"].values.astype(float)

# Remove rows with NaN pIC50
valid_train = np.isfinite(y_train)
valid_test  = np.isfinite(y_test)
smiles_train = [s for s, v in zip(smiles_train, valid_train) if v]
smiles_test  = [s for s, v in zip(smiles_test,  valid_test)  if v]
y_train = y_train[valid_train]
y_test  = y_test[valid_test]
print(f"    After NaN filter: {len(smiles_train)} train, {len(smiles_test)} test")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute descriptors
# ─────────────────────────────────────────────────────────────────────────────
print("[2/5] Computing descriptors (this may take a minute for Mordred)...")
X_train_raw = compute_descriptors(smiles_train)
X_test_raw  = compute_descriptors(smiles_test)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Train all models, select best by R² on Series C
# ─────────────────────────────────────────────────────────────────────────────
print("[3/5] Training models across all descriptor sets...")
best_r2 = -np.inf
best_bundle = None
results = []

for desc_name, X_tr_raw in X_train_raw.items():
    imp = SimpleImputer(strategy="mean")
    X_tr = imp.fit_transform(X_tr_raw)
    X_te = imp.transform(X_test_raw[desc_name])

    for model_name, factory in model_factories.items():
        X_tr_m, X_te_m = X_tr.copy(), X_te.copy()
        sc = None
        if model_name in scaled_models:
            sc = StandardScaler()
            X_tr_m = sc.fit_transform(X_tr_m)
            X_te_m = sc.transform(X_te_m)
        try:
            model = factory({})
            model.fit(X_tr_m, y_train)
            y_pred = model.predict(X_te_m)
            r2  = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
        except Exception as exc:
            print(f"    SKIP {desc_name}/{model_name}: {exc}")
            continue

        results.append({"Descriptor": desc_name, "Model": model_name, "R2": r2, "MAE": mae})

        if r2 > best_r2:
            best_r2 = r2
            best_bundle = {
                "descriptor_name": desc_name,
                "imputer": imp,
                "scaler": sc,
                "model": model,
            }
            print(f"    ★ New best: {desc_name}/{model_name}  R²={r2:.4f}  MAE={mae:.4f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print(f"\n[4/5] Top 10 models by R²:")
print(results_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Save artifact
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
artifact_path = OUT_DIR / "qsar_best_model.joblib"

if best_bundle is None:
    raise RuntimeError("No valid model was trained. Check data and descriptor imports.")

joblib.dump(best_bundle, artifact_path)
print(f"\n[5/5] Saved → {artifact_path}")
print(f"       Descriptor : {best_bundle['descriptor_name']}")
print(f"       Model      : {type(best_bundle['model']).__name__}")
print(f"       Target     : log10(IC50_uM)  [scorer: pIC50 = 6 - prediction]")
print(f"       R² (Ser C) : {best_r2:.4f}")

# Also save a full results CSV
results_csv = OUT_DIR / "model_selection_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"       All results: {results_csv}")
