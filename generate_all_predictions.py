#!/usr/bin/env python
"""
Generate predictions for all data subsets: Series E, Series C test, dropped rows, and non-numeric IC50 rows.
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(42)

print("\n" + "="*80)
print("GENERATING PREDICTIONS FOR ALL DATA SUBSETS")
print("="*80)

# Change to workspace directory
os.chdir("/Users/nb/Documents/Tzu-qsar-generative-model")

# Import pipeline modules
from data_loader import (load_excel_sheets, apply_smiles_cleaning,
                         combine_and_deduplicate, filter_numeric_ic50,
                         standardize_smiles)
from descriptors import compute_descriptors
from model import train_and_select, predict_and_antilog

print("\n[1/5] Loading and preprocessing data...")
series_dfs = load_excel_sheets("TB Project QSAR.xlsx")
series_dfs = apply_smiles_cleaning(series_dfs)
df = combine_and_deduplicate(series_dfs)

# Separate into numeric and non-numeric
unique_df = df.drop_duplicates(subset=['Canonical_SMILES']).copy()
non_numeric_rows = unique_df[pd.to_numeric(unique_df["IC50 uM"], errors="coerce").isna()].copy()
numeric_df = unique_df[pd.to_numeric(unique_df["IC50 uM"], errors="coerce").notna()].copy()

# Extract dropped rows (IC50 = 100, 200)
dropped_rows = numeric_df[numeric_df["IC50 uM"].isin([100, 200])].copy()
numeric_df = numeric_df.drop(dropped_rows.index)

# Log transform
numeric_df['transformed_IC50'] = np.log10(numeric_df['IC50 uM'] + 1e-8)

print(f"  ✓ Total molecules: {len(unique_df)}")
print(f"  ✓ Numeric IC50: {len(numeric_df)}")
print(f"  ✓ Dropped rows (IC50=100,200): {len(dropped_rows)}")
print(f"  ✓ Non-numeric IC50: {len(non_numeric_rows)}")

print("\n[2/5] Standardizing SMILES...")
numeric_df['cleanedMol'] = numeric_df['Canonical_SMILES'].apply(
    lambda x: standardize_smiles(x, verbose=False)
)
dropped_rows['cleanedMol'] = dropped_rows['Canonical_SMILES'].apply(
    lambda x: standardize_smiles(x, verbose=False)
)
non_numeric_rows['cleanedMol'] = non_numeric_rows['Canonical_SMILES'].apply(
    lambda x: standardize_smiles(x, verbose=False)
)
print(f"  ✓ SMILES standardization complete")

print("\n[3/5] Preparing data splits...")
train_df = numeric_df[numeric_df["Series_Code"].isin(["A","B","D"])].reset_index(drop=True)
test_df = numeric_df[numeric_df["Series_Code"] == "C"].reset_index(drop=True)
predict_df = numeric_df[numeric_df["Series_Code"] == "E"].reset_index(drop=True)

print(f"  ✓ Training set: {len(train_df)} molecules")
print(f"  ✓ Test set: {len(test_df)} molecules")
print(f"  ✓ Prediction set: {len(predict_df)} molecules")

print("\n[4/5] Training best model...")
X_train_raw = compute_descriptors(train_df["cleanedMol"].values)
X_test_raw = compute_descriptors(test_df["cleanedMol"].values)
X_predict_raw = compute_descriptors(predict_df["cleanedMol"].values)
X_dropped_raw = compute_descriptors(dropped_rows["cleanedMol"].values)
X_non_numeric_raw = compute_descriptors(non_numeric_rows["cleanedMol"].values)

y_train = train_df["transformed_IC50"].values
y_test = test_df["transformed_IC50"].values

trained_models, imputers, scalers, best_desc, best_model_name, results_df = \
    train_and_select(X_train_raw, y_train, X_test_raw, y_test)

print(f"  ✓ Best model: {best_model_name} using {best_desc}")
print(f"  ✓ Best R² score: {results_df['R2'].max():.4f}")

print("\n[5/5] Making predictions for all data subsets...")

# Get best model components
model_best = trained_models[(best_desc, best_model_name)]
imp_best = imputers[best_desc]
sc_best = scalers.get((best_desc, best_model_name), None)

# Transform data
X_predict_best = imp_best.transform(X_predict_raw[best_desc])
X_test_best = imp_best.transform(X_test_raw[best_desc])
X_dropped_best = imp_best.transform(X_dropped_raw[best_desc])
X_non_numeric_best = imp_best.transform(X_non_numeric_raw[best_desc])

if sc_best:
    X_predict_best = sc_best.transform(X_predict_best)
    X_test_best = sc_best.transform(X_test_best)
    X_dropped_best = sc_best.transform(X_dropped_best)
    X_non_numeric_best = sc_best.transform(X_non_numeric_best)

# Make predictions
pred_E_df = predict_and_antilog(model_best, X_predict_best, predict_df)
pred_test_df = predict_and_antilog(model_best, X_test_best, test_df)
pred_dropped_df = predict_and_antilog(model_best, X_dropped_best, dropped_rows)
pred_non_numeric_df = predict_and_antilog(model_best, X_non_numeric_best, non_numeric_rows)

# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Save all predictions
pred_E_df.to_csv(os.path.join(output_dir, "SeriesE_with_predictions.csv"), index=False)
print(f"  ✓ Saved {len(pred_E_df)} Series E predictions")

pred_test_df.to_csv(os.path.join(output_dir, "SeriesC_test_with_predictions.csv"), index=False)
print(f"  ✓ Saved {len(pred_test_df)} Series C test predictions")

pred_dropped_df.to_csv(os.path.join(output_dir, "Dropped_rows_predictions.csv"), index=False)
print(f"  ✓ Saved {len(pred_dropped_df)} dropped rows predictions")

pred_non_numeric_df.to_csv(os.path.join(output_dir, "Non_numeric_rows_predictions.csv"), index=False)
print(f"  ✓ Saved {len(pred_non_numeric_df)} non-numeric IC50 predictions")

# Summary
print("\n" + "="*80)
print("PREDICTION GENERATION COMPLETE")
print("="*80)
print(f"\n📊 Output Files Created:")
print(f"  1. SeriesE_with_predictions.csv ({len(pred_E_df)} rows)")
print(f"  2. SeriesC_test_with_predictions.csv ({len(pred_test_df)} rows)")
print(f"  3. Dropped_rows_predictions.csv ({len(pred_dropped_df)} rows)")
print(f"  4. Non_numeric_rows_predictions.csv ({len(pred_non_numeric_df)} rows)")
print(f"\n📌 All files located in: {os.path.abspath(output_dir)}/")
print(f"📌 All files include 'cleanedMol' column with standardized SMILES")
print(f"📌 Model used: {best_model_name} with {best_desc} descriptors")
print("\n" + "="*80 + "\n")
