"""
main.py
Main script to run the QSAR pipeline using modularized code.
"""
import os
import pandas as pd
import numpy as np
from data_loader import load_excel_sheets, apply_smiles_cleaning, combine_and_deduplicate, filter_numeric_ic50
from descriptors import compute_descriptors
from clustering import run_umap, run_hdbscan, get_scaffold_safe
from model import train_and_select, predict_and_antilog
from visualization import plot_ic50_distribution, plot_umap_clusters, plot_y_true_vs_pred

def main():
    # 1. Load data
        print("[1/7] Loading data...")
    file_name = "your_data_file.xlsx"  # <-- Set your file name here
    series_dfs = load_excel_sheets(file_name)
        print("[2/7] Cleaning SMILES and deduplicating...")
    series_dfs = apply_smiles_cleaning(series_dfs)
    df = combine_and_deduplicate(series_dfs)
    numeric_df = filter_numeric_ic50(df)

    # 2. Univariate analysis
        print("[3/7] Univariate analysis and plotting...")
    output_dir = "outputs"
    import os
    os.makedirs(output_dir, exist_ok=True)
    # Save IC50 distribution plot
    import matplotlib.pyplot as plt
    plt.ioff()
    plot_ic50_distribution(numeric_df["IC50 uM"].dropna())
    plt.savefig(os.path.join(output_dir, "ic50_distribution.png"))
    plt.close()
    numeric_df['transformed_IC50'] = np.log10(numeric_df['IC50 uM'] + 1e-8)

    # 3. Descriptor calculation
        print("[4/7] Calculating descriptors...")
    smiles_list = numeric_df['Canonical_SMILES'].tolist()
    desc_dict = compute_descriptors(smiles_list)

    # 4. UMAP + HDBSCAN clustering
        print("[5/7] Running UMAP dimensionality reduction (this may take a while)...")
    descriptor_matrix = desc_dict['RDKit']
    umap_result = run_umap(descriptor_matrix)
    numeric_df['UMAP1'] = umap_result[:, 0]
    numeric_df['UMAP2'] = umap_result[:, 1]
        print("[6/7] Running HDBSCAN clustering (this may take a while)...")
    clusters, min_size, score = run_hdbscan(umap_result)
    numeric_df['Cluster'] = clusters
    numeric_df['Scaffold'] = numeric_df['Canonical_SMILES'].apply(lambda s: get_scaffold_safe(Chem.MolFromSmiles(s)))
    plot_umap_clusters(numeric_df)
    plt.savefig(os.path.join(output_dir, "umap_clusters.png"))
    plt.close()

    # 5. Model training and evaluation
        print("[7/7] Model training, evaluation, and saving outputs...")
    # Example split (adjust as needed)
    train_df = numeric_df[numeric_df["Series_Code"].isin(["A","B","D"])].reset_index(drop=True)
    test_df = numeric_df[numeric_df["Series_Code"] == "C"].reset_index(drop=True)
    predict_df = numeric_df[numeric_df["Series_Code"] == "E"].reset_index(drop=True)
    smiles_train = train_df["Canonical_SMILES"].values
    y_train = train_df["transformed_IC50"].values
    smiles_test = test_df["Canonical_SMILES"].values
    y_test = test_df["transformed_IC50"].values
    smiles_predict = predict_df["Canonical_SMILES"].values
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
    y_test_orig = 10 ** y_test
    y_test_pred_best = pred_test_df["Predicted_IC50"].values
    plot_y_true_vs_pred(y_test_orig, y_test_pred_best, best_model_name, best_desc)
    plt.savefig(os.path.join(output_dir, "y_true_vs_pred.png"))
    plt.close()
    # Save outputs to outputs/ directory
    pred_E_df.to_csv(os.path.join(output_dir, "SeriesE_with_predictions.csv"), index=False)
    pred_test_df.to_csv(os.path.join(output_dir, "SeriesC_test_with_predictions.csv"), index=False)
    # If you have dropped_rows or non_numeric_rows, save them as well if they exist
    if 'dropped_rows' in locals() and not dropped_rows.empty:
        pred_dropped_df = predict_and_antilog(model_best, X_test_best, dropped_rows)
        pred_dropped_df.to_csv(os.path.join(output_dir, "Dropped_rows_predictions.csv"), index=False)
    if 'non_numeric_rows' in locals() and not non_numeric_rows.empty:
        pred_non_numeric_df = predict_and_antilog(model_best, X_test_best, non_numeric_rows)
        pred_non_numeric_df.to_csv(os.path.join(output_dir, "Non_numeric_rows_predictions.csv"), index=False)
    print(f"Pipeline complete. All outputs (CSVs and plots) saved in '{output_dir}/'.")

if __name__ == "__main__":
    main()
