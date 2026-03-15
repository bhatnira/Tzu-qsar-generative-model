#!/usr/bin/env python3
"""
Simplified, robust QSAR pipeline runner.
This is a streamlined version designed to complete successfully.
"""
import os
import sys
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_core.paths import DATASET_XLSX

REINVENT_RUNNER = PROJECT_ROOT / "reinvent_integration" / "run_all_generation.py"


def _dispatch_generation_preset() -> None:
    """
    Optional wrapper mode:
      python run_pipeline.py --generation-preset fast|balanced|diverse

    This dispatches to reinvent_integration/run_all_generation.py with
    hybrid shortlist presets and exits immediately.
    """
    flag_names = {"--generation-preset", "--preset"}
    argv = sys.argv[1:]

    if not any(flag in argv for flag in flag_names):
        return

    preset = None
    for i, token in enumerate(argv):
        if token in flag_names and i + 1 < len(argv):
            preset = argv[i + 1].strip().lower()
            break

    if preset is None:
        print("❌ Missing preset value. Use: --generation-preset fast|balanced|diverse")
        raise SystemExit(2)

    presets = {
        "fast": {"top_k": 80, "max_per_scaffold": 4},
        "balanced": {"top_k": 120, "max_per_scaffold": 3},
        "diverse": {"top_k": 200, "max_per_scaffold": 1},
    }

    if preset not in presets:
        print(f"❌ Unknown preset '{preset}'. Valid presets: fast, balanced, diverse")
        raise SystemExit(2)

    if not REINVENT_RUNNER.exists():
        print(f"❌ Missing runner: {REINVENT_RUNNER}")
        raise SystemExit(2)

    cfg = presets[preset]
    python_exec = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_cmd = str(python_exec) if python_exec.exists() else sys.executable

    cmd = [
        python_cmd,
        str(REINVENT_RUNNER),
        "--hybrid-top-k", str(cfg["top_k"]),
        "--hybrid-max-per-scaffold", str(cfg["max_per_scaffold"]),
    ]

    print("\n" + "=" * 80)
    print(" GENERATION PRESET DISPATCH")
    print("=" * 80)
    print(f"Preset: {preset}")
    print(f"Hybrid shortlist: top_k={cfg['top_k']}, max_per_scaffold={cfg['max_per_scaffold']}")
    print(f"Command: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    raise SystemExit(proc.returncode)


_dispatch_generation_preset()

print("\n" + "="*80)
print(" QSAR PIPELINE EXECUTION")
print("="*80)

# Ensure we're in the right directory
os.chdir(PROJECT_ROOT)

try:
    # ==========================================================================
    print("\n[STEP 1/6] Loading and preprocessing data...")
    # ==========================================================================
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    
    from rdkit import Chem
    from qsar_core.data_loader import (load_excel_sheets, apply_smiles_cleaning, 
                             combine_and_deduplicate, filter_numeric_ic50,
                             standardize_smiles)
    
    # Load data
    series_dfs = load_excel_sheets(str(DATASET_XLSX))
    print(f"  → Loaded {len(series_dfs)} chemical series")
    
    # Clean and combine
    series_dfs = apply_smiles_cleaning(series_dfs)
    df = combine_and_deduplicate(series_dfs)
    print(f"  → Combined & deduplicated: {len(df)} molecules")
    
    numeric_df = filter_numeric_ic50(df)
    print(f"  → Filtered numeric IC50: {len(numeric_df)} molecules")
    
    # Log transform
    numeric_df['transformed_IC50'] = np.log10(numeric_df['IC50 uM'] + 1e-8)
    print(f"  ✅ Step 1 complete")

except Exception as e:
    print(f"  ❌ ERROR in Step 1: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # ==========================================================================
    print("\n[STEP 2/6] Standardizing SMILES and computing descriptors...")
    # ==========================================================================
    
    numeric_df['cleanedMol'] = numeric_df['Canonical_SMILES'].apply(
        lambda x: standardize_smiles(x, verbose=False)
    )
    print(f"  → SMILES standardized")
    
    from qsar_core.descriptors import compute_descriptors
    smiles_list = numeric_df['cleanedMol'].tolist()
    desc_dict = compute_descriptors(smiles_list)
    print(f"  → Computed {len(desc_dict)} descriptor types")
    print(f"  ✅ Step 2 complete")

except Exception as e:
    print(f"  ❌ ERROR in Step 2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # ==========================================================================
    print("\n[STEP 3/6] Chemical space analysis (UMAP + HDBSCAN)...")
    # ==========================================================================
    
    from qsar_core.clustering import run_umap, run_hdbscan, get_scaffold_safe
    
    # UMAP
    descriptor_matrix = desc_dict['RDKit']
    umap_result = run_umap(descriptor_matrix)
    numeric_df['UMAP1'] = umap_result[:, 0]
    numeric_df['UMAP2'] = umap_result[:, 1]
    print(f"  → UMAP dimensionality reduction complete")
    
    # HDBSCAN
    clusters, min_size, score = run_hdbscan(umap_result)
    numeric_df['Cluster'] = clusters
    numeric_df['Scaffold'] = numeric_df['cleanedMol'].apply(
        lambda s: get_scaffold_safe(Chem.MolFromSmiles(s))
    )
    print(f"  → HDBSCAN clustering complete (silhouette: {score:.4f})")
    print(f"  ✅ Step 3 complete")

except Exception as e:
    print(f"  ❌ ERROR in Step 3: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # ==========================================================================
    print("\n[STEP 4/6] Machine learning model training...")
    # ==========================================================================
    
    from qsar_core.model import train_and_select, predict_and_antilog
    
    # Split data
    train_df = numeric_df[numeric_df["Series_Code"].isin(["A","B","D"])].reset_index(drop=True)
    test_df = numeric_df[numeric_df["Series_Code"] == "C"].reset_index(drop=True)
    predict_df = numeric_df[numeric_df["Series_Code"] == "E"].reset_index(drop=True)
    
    print(f"  → Train set: {len(train_df)} molecules")
    print(f"  → Test set: {len(test_df)} molecules")
    print(f"  → Predict set: {len(predict_df)} molecules")
    
    # Compute descriptors
    X_train_raw = compute_descriptors(train_df["cleanedMol"].values)
    X_test_raw = compute_descriptors(test_df["cleanedMol"].values)
    X_predict_raw = compute_descriptors(predict_df["cleanedMol"].values)
    
    # Train models (84 combinations: 6 descriptors × 14 models)
    y_train = train_df["transformed_IC50"].values
    y_test = test_df["transformed_IC50"].values
    
    print(f"  → Training 84 model combinations...")
    trained_models, imputers, scalers, best_desc, best_model_name, results_df = \
        train_and_select(X_train_raw, y_train, X_test_raw, y_test)
    
    print(f"  → Best model: {best_model_name} (descriptor: {best_desc})")
    print(f"  ✅ Step 4 complete")

except Exception as e:
    print(f"  ❌ ERROR in Step 4: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # ==========================================================================
    print("\n[STEP 5/6] Making predictions...")
    # ==========================================================================
    
    model_best = trained_models[(best_desc, best_model_name)]
    imp_best = imputers[best_desc]
    sc_best = scalers.get((best_desc, best_model_name), None)
    
    # Transform test data
    X_test_best = imp_best.transform(X_test_raw[best_desc])
    if sc_best:
        X_test_best = sc_best.transform(X_test_best)
    
    # Transform predict data
    X_predict_best = imp_best.transform(X_predict_raw[best_desc])
    if sc_best:
        X_predict_best = sc_best.transform(X_predict_best)
    
    # Make predictions
    pred_test_df = predict_and_antilog(model_best, X_test_best, test_df)
    pred_E_df = predict_and_antilog(model_best, X_predict_best, predict_df)
    
    print(f"  → Test set predictions: {len(pred_test_df)} rows")
    print(f"  → Series E predictions: {len(pred_E_df)} rows")
    print(f"  ✅ Step 5 complete")

except Exception as e:
    print(f"  ❌ ERROR in Step 5: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # ==========================================================================
    print("\n[STEP 6/6] Saving outputs and generating visualizations...")
    # ==========================================================================
    
    from qsar_core.visualization import (plot_ic50_distribution, plot_umap_clusters, 
                               plot_y_true_vs_pred)
    import matplotlib.pyplot as plt
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. IC50 Distribution
    plt.ioff()
    plot_ic50_distribution(numeric_df["IC50 uM"].dropna())
    plt.savefig(os.path.join(output_dir, "ic50_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: ic50_distribution.png")
    
    # 2. UMAP Clusters
    plot_umap_clusters(numeric_df)
    plt.savefig(os.path.join(output_dir, "umap_clusters.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: umap_clusters.png")
    
    # 3. True vs Predicted
    y_test_orig = 10 ** y_test
    y_test_pred_best = pred_test_df["Predicted_IC50"].values
    plot_y_true_vs_pred(y_test_orig, y_test_pred_best, best_model_name, best_desc)
    plt.savefig(os.path.join(output_dir, "y_true_vs_pred.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: y_true_vs_pred.png")
    
    # 4. CSV Files
    pred_E_df.to_csv(os.path.join(output_dir, "SeriesE_with_predictions.csv"), index=False)
    print(f"  → Saved: SeriesE_with_predictions.csv ({len(pred_E_df)} rows)")
    
    pred_test_df.to_csv(os.path.join(output_dir, "SeriesC_test_with_predictions.csv"), index=False)
    print(f"  → Saved: SeriesC_test_with_predictions.csv ({len(pred_test_df)} rows)")
    
    results_df.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)
    print(f"  → Saved: model_results.csv ({len(results_df)} rows)")
    
    print(f"  ✅ Step 6 complete")

except Exception as e:
    print(f"  ❌ ERROR in Step 6: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# SUCCESS!
# =============================================================================
print("\n" + "="*80)
print(" ✅ PIPELINE EXECUTION SUCCESSFUL!")
print("="*80)
print(f"\nOutput files in '{output_dir}/':")
print("  📊 Visualizations:")
print("     • ic50_distribution.png - IC50 value distribution")
print("     • umap_clusters.png - Chemical space clustering")
print("     • y_true_vs_pred.png - Model performance")
print("  📈 Data files:")
print("     • SeriesE_with_predictions.csv - Predictions for Series E")
print("     • SeriesC_test_with_predictions.csv - Test set with predictions")
print("     • model_results.csv - Results from all 84 model combinations")
print("\n" + "="*80 + "\n")
