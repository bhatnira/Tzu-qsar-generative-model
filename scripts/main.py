"""
main.py
Main script to run the QSAR pipeline using modularized code.
"""
import os
import sys
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_core.paths import DATASET_XLSX
from qsar_core.data_loader import load_excel_sheets, apply_smiles_cleaning, combine_and_deduplicate, filter_numeric_ic50
from qsar_core.descriptors import compute_descriptors
from qsar_core.clustering import run_umap, run_hdbscan, get_scaffold_safe
from qsar_core.model import train_and_select, predict_and_antilog
from qsar_core.visualization import plot_ic50_distribution, plot_umap_clusters, plot_y_true_vs_pred


def _canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _predict_dataframe(df_in, best_desc, model_best, imp_best, sc_best):
    df_out = df_in.copy()
    if df_out.empty:
        df_out["Predicted_IC50"] = []
        df_out["Predicted_IC50_uM"] = []
        df_out["Predicted_pIC50"] = []
        return df_out

    smiles_values = df_out["Canonical_SMILES"].astype(str).values
    x_raw = compute_descriptors(smiles_values)
    x_best = imp_best.transform(x_raw[best_desc])
    if sc_best is not None:
        x_best = sc_best.transform(x_best)

    pred_df = predict_and_antilog(model_best, x_best, df_out)
    pred_df["Predicted_IC50_uM"] = pred_df["Predicted_IC50"]
    pred_df["Predicted_pIC50"] = 6.0 - np.log10(pred_df["Predicted_IC50_uM"] + 1e-12)
    return pred_df

def main():
    # 1. Load data
    print("[1/7] Loading data...")
    file_name = str(DATASET_XLSX)
    series_dfs = load_excel_sheets(file_name)
    combined_raw = pd.concat(series_dfs, ignore_index=True)
    print("[2/7] Cleaning SMILES and deduplicating...")
    series_dfs = apply_smiles_cleaning(series_dfs)
    df = combine_and_deduplicate(series_dfs)
    numeric_df = filter_numeric_ic50(df)

    combined_raw = pd.concat(series_dfs, ignore_index=True)
    if "Clean_SMILES" in combined_raw.columns:
        combined_raw["Canonical_SMILES"] = combined_raw["Clean_SMILES"].apply(_canonicalize_smiles)
    else:
        combined_raw["Canonical_SMILES"] = None

    # 2. Setup output directory
    print("[3/7] Setting up output directory and univariate analysis...")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    # Save IC50 distribution plot
    import matplotlib.pyplot as plt
    plt.ioff()
    plot_ic50_distribution(numeric_df["IC50 uM"].dropna())
    plt.savefig(os.path.join(output_dir, "ic50_distribution.png"))
    plt.close()
    numeric_df['transformed_IC50'] = np.log10(numeric_df['IC50 uM'] + 1e-8)

    # 3. Standardize SMILES and descriptor calculation
    print("[4/7] Standardizing SMILES and calculating descriptors...")
    from qsar_core.data_loader import standardize_smiles
    numeric_df['cleanedMol'] = numeric_df['Canonical_SMILES'].apply(
        lambda x: standardize_smiles(x, verbose=False)
    )
    smiles_list = numeric_df['cleanedMol'].tolist()
    desc_dict = compute_descriptors(smiles_list)

    # 4. UMAP + HDBSCAN clustering
    print("[5/7] Running UMAP dimensionality reduction...")
    descriptor_matrix = desc_dict['RDKit']
    umap_result = run_umap(descriptor_matrix)
    numeric_df['UMAP1'] = umap_result[:, 0]
    numeric_df['UMAP2'] = umap_result[:, 1]
    print("[6/7] Running HDBSCAN clustering...")
    clusters, min_size, score = run_hdbscan(umap_result)
    numeric_df['Cluster'] = clusters
    numeric_df['Scaffold'] = numeric_df['cleanedMol'].apply(lambda s: get_scaffold_safe(Chem.MolFromSmiles(s)))
    plot_umap_clusters(numeric_df)
    plt.savefig(os.path.join(output_dir, "umap_clusters.png"))
    plt.close()

    # 5. Model training and evaluation
    print("[7/7] Model training, evaluation, and saving outputs...")
    train_df = numeric_df[numeric_df["Series_Code"].isin(["A","B","D"])].reset_index(drop=True)
    test_df = numeric_df[numeric_df["Series_Code"] == "C"].reset_index(drop=True)
    predict_df = numeric_df[numeric_df["Series_Code"] == "E"].reset_index(drop=True)
    smiles_train = train_df["cleanedMol"].values
    y_train = train_df["transformed_IC50"].values
    smiles_test = test_df["cleanedMol"].values
    y_test = test_df["transformed_IC50"].values
    smiles_predict = predict_df["cleanedMol"].values
    X_train_raw = compute_descriptors(smiles_train)
    X_test_raw = compute_descriptors(smiles_test)
    X_predict_raw = compute_descriptors(smiles_predict)
    trained_models, imputers, scalers, best_desc, best_model_name, results_df = train_and_select(X_train_raw, y_train, X_test_raw, y_test)
    model_best = trained_models[(best_desc, best_model_name)]
    imp_best = imputers[best_desc]
    sc_best = scalers.get((best_desc, best_model_name), None)
    X_test_best = imp_best.transform(X_test_raw[best_desc])
    X_predict_best = imp_best.transform(X_predict_raw[best_desc])
    if sc_best:
        X_test_best = sc_best.transform(X_test_best)
        X_predict_best = sc_best.transform(X_predict_best)
    pred_test_df = predict_and_antilog(model_best, X_test_best, test_df)
    pred_E_df = predict_and_antilog(model_best, X_predict_best, predict_df)
    pred_test_df["Predicted_IC50_uM"] = pred_test_df["Predicted_IC50"]
    pred_test_df["Predicted_pIC50"] = 6.0 - np.log10(pred_test_df["Predicted_IC50_uM"] + 1e-12)
    pred_E_df["Predicted_IC50_uM"] = pred_E_df["Predicted_IC50"]
    pred_E_df["Predicted_pIC50"] = 6.0 - np.log10(pred_E_df["Predicted_IC50_uM"] + 1e-12)
    y_test_orig = 10 ** y_test
    y_test_pred_best = pred_test_df["Predicted_IC50"].values
    plot_y_true_vs_pred(y_test_orig, y_test_pred_best, best_model_name, best_desc)
    plt.savefig(os.path.join(output_dir, "y_true_vs_pred.png"))
    plt.close()
    # Save outputs to outputs/ directory
    pred_E_df.to_csv(os.path.join(output_dir, "SeriesE_with_predictions.csv"), index=False)
    pred_test_df.to_csv(os.path.join(output_dir, "SeriesC_test_with_predictions.csv"), index=False)

    # Save model selection table
    results_df.to_csv(os.path.join(output_dir, "model_selection_results.csv"), index=False)

    # Predict additional rows
    non_numeric_mask = pd.to_numeric(combined_raw["IC50 uM"], errors="coerce").isna()
    non_numeric_df = combined_raw[non_numeric_mask].copy()
    non_numeric_df = non_numeric_df.dropna(subset=["Canonical_SMILES"]).reset_index(drop=True)
    pred_non_numeric_df = _predict_dataframe(non_numeric_df, best_desc, model_best, imp_best, sc_best)
    pred_non_numeric_df.to_csv(os.path.join(output_dir, "Non_numeric_rows_predictions.csv"), index=False)

    dedup_key = df[["Canonical_SMILES"]].drop_duplicates().copy()
    dropped_df = combined_raw.dropna(subset=["Canonical_SMILES"]).copy()
    dropped_df = dropped_df.merge(dedup_key.assign(_kept=1), on="Canonical_SMILES", how="left")
    dropped_df = dropped_df[dropped_df["_kept"].isna()].drop(columns=["_kept"]).reset_index(drop=True)
    pred_dropped_df = _predict_dataframe(dropped_df, best_desc, model_best, imp_best, sc_best)
    pred_dropped_df.to_csv(os.path.join(output_dir, "Dropped_rows_predictions.csv"), index=False)

    manifest_df = pd.DataFrame(
        [
            {"file": "SeriesC_test_with_predictions.csv", "rows": len(pred_test_df)},
            {"file": "SeriesE_with_predictions.csv", "rows": len(pred_E_df)},
            {"file": "model_selection_results.csv", "rows": len(results_df)},
        ]
    )
    manifest_df.to_csv(os.path.join(output_dir, "qsar_outputs_manifest.csv"), index=False)

    additional_manifest_df = pd.DataFrame(
        [
            {"file": "Non_numeric_rows_predictions.csv", "rows": len(pred_non_numeric_df)},
            {"file": "Dropped_rows_predictions.csv", "rows": len(pred_dropped_df)},
        ]
    )
    additional_manifest_df.to_csv(os.path.join(output_dir, "additional_qsar_outputs_manifest.csv"), index=False)

    print(f"Pipeline complete. All outputs (CSVs and plots) saved in '{output_dir}/'.")

if __name__ == "__main__":
    main()
