"""
debug_main.py
Debug version of main.py with detailed error reporting.
"""
import os
import sys
import traceback
import pandas as pd
import numpy as np
from rdkit import Chem

print("=" * 70)
print("QSAR PIPELINE - DEBUG MODE")
print("=" * 70)

try:
    print("\n[1/7] Loading modules...")
    from data_loader import load_excel_sheets, apply_smiles_cleaning, combine_and_deduplicate, filter_numeric_ic50, standardize_smiles
    print("✅ data_loader imported")
    
    from descriptors import compute_descriptors
    print("✅ descriptors imported")
    
    from clustering import run_umap, run_hdbscan, get_scaffold_safe
    print("✅ clustering imported")
    
    from model import train_and_select, predict_and_antilog
    print("✅ model imported")
    
    from visualization import plot_ic50_distribution, plot_umap_clusters, plot_y_true_vs_pred
    print("✅ visualization imported")
    
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[2/7] Loading data...")
    file_name = "TB Project QSAR.xlsx"
    series_dfs = load_excel_sheets(file_name)
    print(f"✅ Loaded {len(series_dfs)} series")
    
    print("\n[3/7] Cleaning SMILES and deduplicating...")
    series_dfs = apply_smiles_cleaning(series_dfs)
    print("✅ SMILES cleaned")
    
    df = combine_and_deduplicate(series_dfs)
    print(f"✅ Combined: {len(df)} molecules (duplicates removed)")
    
    numeric_df = filter_numeric_ic50(df)
    print(f"✅ Filtered: {len(numeric_df)} molecules with numeric IC50")
    
except Exception as e:
    print(f"❌ Data loading error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[4/7] Setting up output directory...")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory ready: {output_dir}/")
    
    print("\n[5/7] Plotting IC50 distribution...")
    plt.ioff()
    plot_ic50_distribution(numeric_df["IC50 uM"].dropna())
    plt.savefig(os.path.join(output_dir, "ic50_distribution.png"))
    plt.close()
    print("✅ IC50 distribution saved")
    
    numeric_df['transformed_IC50'] = np.log10(numeric_df['IC50 uM'] + 1e-8)
    print("✅ IC50 log-transformed")
    
except Exception as e:
    print(f"❌ Visualization error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[6/7] Standardizing SMILES and computing descriptors...")
    numeric_df['cleanedMol'] = numeric_df['Canonical_SMILES'].apply(
        lambda x: standardize_smiles(x, verbose=False)
    )
    print(f"✅ SMILES standardized")
    
    smiles_list = numeric_df['cleanedMol'].tolist()
    desc_dict = compute_descriptors(smiles_list)
    print(f"✅ Computed descriptors: {list(desc_dict.keys())}")
    
except Exception as e:
    print(f"❌ Descriptor calculation error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[7/7] Running UMAP and HDBSCAN clustering...")
    descriptor_matrix = desc_dict['RDKit']
    print(f"  Descriptor matrix shape: {descriptor_matrix.shape}")
    
    umap_result = run_umap(descriptor_matrix)
    print(f"✅ UMAP completed: {umap_result.shape}")
    
    numeric_df['UMAP1'] = umap_result[:, 0]
    numeric_df['UMAP2'] = umap_result[:, 1]
    
    clusters, min_size, score = run_hdbscan(umap_result)
    print(f"✅ HDBSCAN completed: min_size={min_size}, silhouette={score:.4f}")
    
    numeric_df['Cluster'] = clusters
    numeric_df['Scaffold'] = numeric_df['cleanedMol'].apply(lambda s: get_scaffold_safe(Chem.MolFromSmiles(s)))
    
    plot_umap_clusters(numeric_df)
    plt.savefig(os.path.join(output_dir, "umap_clusters.png"))
    plt.close()
    print("✅ UMAP clusters saved")
    
except Exception as e:
    print(f"❌ Clustering error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[8/7] Training machine learning models...")
    train_df = numeric_df[numeric_df["Series_Code"].isin(["A","B","D"])].reset_index(drop=True)
    test_df = numeric_df[numeric_df["Series_Code"] == "C"].reset_index(drop=True)
    predict_df = numeric_df[numeric_df["Series_Code"] == "E"].reset_index(drop=True)
    
    print(f"  Train set: {len(train_df)} molecules")
    print(f"  Test set: {len(test_df)} molecules")
    print(f"  Predict set: {len(predict_df)} molecules")
    
    smiles_train = train_df["cleanedMol"].values
    y_train = train_df["transformed_IC50"].values
    smiles_test = test_df["cleanedMol"].values
    y_test = test_df["transformed_IC50"].values
    smiles_predict = predict_df["cleanedMol"].values
    
    print("  Computing descriptors for train/test/predict sets...")
    X_train_raw = compute_descriptors(smiles_train)
    X_test_raw = compute_descriptors(smiles_test)
    X_predict_raw = compute_descriptors(smiles_predict)
    print("✅ Descriptors computed")
    
    print("  Training 84 model combinations (6 descriptors × 14 models)...")
    print("  This may take several minutes...")
    trained_models, imputers, scalers, best_desc, best_model_name, results_df = train_and_select(X_train_raw, y_train, X_test_raw, y_test)
    print(f"✅ Best model: {best_model_name} with {best_desc}")
    
except Exception as e:
    print(f"❌ Model training error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[9/7] Making predictions...")
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
    print("✅ Predictions completed")
    
    y_test_orig = 10 ** y_test
    y_test_pred_best = pred_test_df["Predicted_IC50"].values
    
    plot_y_true_vs_pred(y_test_orig, y_test_pred_best, best_model_name, best_desc)
    plt.savefig(os.path.join(output_dir, "y_true_vs_pred.png"))
    plt.close()
    print("✅ True vs Predicted plot saved")
    
except Exception as e:
    print(f"❌ Prediction error: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[10/7] Saving outputs...")
    pred_E_df.to_csv(os.path.join(output_dir, "SeriesE_with_predictions.csv"), index=False)
    print("✅ SeriesE_with_predictions.csv saved")
    
    pred_test_df.to_csv(os.path.join(output_dir, "SeriesC_test_with_predictions.csv"), index=False)
    print("✅ SeriesC_test_with_predictions.csv saved")
    
    results_df.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)
    print("✅ model_results.csv saved")
    
except Exception as e:
    print(f"❌ Output saving error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETE - ALL OUTPUTS SAVED")
print("=" * 70)
print(f"\nOutput files in '{output_dir}/':")
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    size = os.path.getsize(fpath)
    print(f"  ✅ {f:40s} ({size:,} bytes)")
print("\n" + "=" * 70)
