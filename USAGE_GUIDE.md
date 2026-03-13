# QSAR Pipeline - Complete Usage Guide

This guide shows how all modules work together to replicate the notebook functionality.

## Installation & Setup

```bash
# Install dependencies
pip install pandas numpy rdkit umap-learn hdbscan scikit-learn xgboost lightgbm catboost matplotlib seaborn mordred openpyxl scipy

# Optional (for advanced visualization)
pip install alphashape shapely
```

## Complete Pipeline Example

Here's how to run the complete QSAR pipeline using the modularized modules:

### **1. Data Loading & Preprocessing**

```python
import numpy as np
import pandas as pd
from data_loader import *

# Set random seed for reproducibility
np.random.seed(42)

# Load data from Excel file
print("[1/7] Loading data...")
file_name = "TB Project QSAR.xlsx"
series_dfs = load_excel_sheets(file_name)

# Clean SMILES strings
print("[2/7] Cleaning SMILES and deduplicating...")
series_dfs = apply_smiles_cleaning(series_dfs)
df = combine_and_deduplicate(series_dfs)
numeric_df = filter_numeric_ic50(df)

# Advanced SMILES standardization
print("[3/7] Standardizing SMILES...")
numeric_df['cleanedMol'] = numeric_df['Canonical_SMILES'].apply(
    lambda x: standardize_smiles(x, verbose=False)
)

# Log-transform IC50 (for regression)
numeric_df['transformed_IC50'] = np.log10(numeric_df['IC50 uM'] + 1e-8)

print(f"Total molecules: {len(numeric_df)}")
```

### **2. Visualize IC50 Distribution**

```python
from visualization import *

# Univariate analysis
print("[4/7] Analyzing IC50 distribution...")
plot_ic50_distribution(numeric_df["IC50 uM"])
plot_ic50_boxplot(numeric_df["IC50 uM"])
summary_stats = univariate_analysis(numeric_df["IC50 uM"])
```

### **3. Compute Molecular Descriptors**

```python
from descriptors import compute_descriptors

print("[5/7] Computing descriptors...")
smiles_list = numeric_df['cleanedMol'].tolist()
X_raw = compute_descriptors(smiles_list)

print("Descriptor types computed:")
for desc_type, matrix in X_raw.items():
    print(f"  - {desc_type}: shape {matrix.shape}")
```

### **4. Chemical Space Analysis with UMAP + HDBSCAN**

```python
from clustering import *

print("[6/7] Running clustering analysis...")

# Dimensionality reduction
descriptor_matrix = X_raw['RDKit']
umap_result = run_umap(descriptor_matrix)

# Clustering with tuning
clusters, min_size, score = run_hdbscan(umap_result)

# Add to dataframe
numeric_df['UMAP1'] = umap_result[:, 0]
numeric_df['UMAP2'] = umap_result[:, 1]
numeric_df['Cluster'] = clusters

# Extract scaffolds
from rdkit import Chem
numeric_df['Molecule'] = numeric_df['cleanedMol'].apply(
    lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None
)
numeric_df['Scaffold'] = numeric_df['Molecule'].apply(get_scaffold_safe)

print(f"Best HDBSCAN min_cluster_size: {min_size}, silhouette: {score:.3f}")

# Visualize
plot_umap_clusters(numeric_df)

# Chemical space analysis
stats = analyze_chemical_space(numeric_df)
print("\nCluster composition:")
print(stats['cluster_series_percent'].round(2))
```

### **5. Machine Learning Model Training & Selection**

```python
from model import *

print("[7/7] Training models...")

# Dataset split (scaffold-based cross-validation)
train_df = numeric_df[numeric_df["Series_Code"].isin(["A","B","D"])].reset_index(drop=True)
test_df = numeric_df[numeric_df["Series_Code"] == "C"].reset_index(drop=True)
predict_df = numeric_df[numeric_df["Series_Code"] == "E"].reset_index(drop=True)

# Extract SMILES for descriptor computation
smiles_train = train_df["cleanedMol"].values
y_train = train_df["transformed_IC50"].values
smiles_test = test_df["cleanedMol"].values
y_test = test_df["transformed_IC50"].values
smiles_predict = predict_df["cleanedMol"].values

# Compute descriptors for train/test/predict
X_train_raw = compute_descriptors(smiles_train)
X_test_raw = compute_descriptors(smiles_test)
X_predict_raw = compute_descriptors(smiles_predict)

# Train all models and select best
trained_models, imputers, scalers, best_desc, best_model_name, results_df = \
    train_and_select(X_train_raw, y_train, X_test_raw, y_test)

print(f"\n✅ Best model selected: {best_model_name} with {best_desc}")
```

### **6. Make Predictions**

```python
# Get best model
model_best = trained_models[(best_desc, best_model_name)]
imp_best = imputers[best_desc]
sc_best = scalers.get((best_desc, best_model_name), None)

# Transform predict data
X_predict = imp_best.transform(X_predict_raw[best_desc])
if sc_best:
    X_predict = sc_best.transform(X_predict)

# Make predictions and convert from log-space
pred_E_df = predict_and_antilog(model_best, X_predict, predict_df)
pred_E_df.to_csv("outputs/SeriesE_predictions.csv", index=False)

# Test set predictions for visualization
X_test = imp_best.transform(X_test_raw[best_desc])
if sc_best:
    X_test = sc_best.transform(X_test)
pred_test_df = predict_and_antilog(model_best, X_test, test_df)

# Plot results
y_test_orig = 10 ** y_test
y_test_pred = pred_test_df["Predicted_IC50"].values
plot_y_true_vs_pred(y_test_orig, y_test_pred, best_model_name, best_desc)

print("✅ Pipeline complete!")
```

### **7. Export Results**

```python
import os

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Save predictions
pred_E_df.to_csv(os.path.join(output_dir, "SeriesE_predictions.csv"), index=False)
pred_test_df.to_csv(os.path.join(output_dir, "SeriesC_test_predictions.csv"), index=False)

# Save model results
results_df.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)

# Save chemical space analysis
stats['cluster_series_table'].to_csv(os.path.join(output_dir, "cluster_composition.csv"))
numeric_df[['Canonical_SMILES', 'UMAP1', 'UMAP2', 'Cluster', 'Scaffold']].to_csv(
    os.path.join(output_dir, "chemical_space.csv"), index=False
)

print("✅ All results saved to 'outputs' directory")
```

---

## Advanced Usage

### **Custom Descriptor Selection**

```python
# Use only specific descriptors
X_train_custom = compute_descriptors(smiles_train)
X_train_subset = {k: X_train_custom[k] for k in ['RDKit', 'ECFP_r2']}

# Train with subset
trained_models, imputers, scalers, best_desc, best_model_name, results_df = \
    train_and_select(X_train_subset, y_train, X_test_subset, y_test)
```

### **Adjusting Clustering Parameters**

```python
# Finer UMAP tuning
umap_result = run_umap(
    descriptor_matrix,
    n_neighbors=50,    # More neighbors for denser clustering
    min_dist=0.05,     # Smaller min_dist for tighter clusters
    n_components=2
)

# Test different HDBSCAN parameters
clusters, min_size, score = run_hdbscan(
    umap_result,
    min_sizes=[5, 10, 15, 20, 30, 50]  # More granular tuning
)
```

### **Custom Model Configuration**

```python
from model import model_factories, scaled_models

# Add custom model
model_factories['CustomSVR'] = lambda p: SVR(C=1.0, kernel='rbf', gamma='scale')
scaled_models.add('CustomSVR')

# Now train_and_select will include your custom model
```

### **Scaffold-Based Cross-Validation**

```python
# 5-fold cross-validation by scaffold

unique_scaffolds = numeric_df['Scaffold'].unique()

for test_scaffold in unique_scaffolds:
    # Test on one scaffold, train on others
    test_mask = numeric_df['Scaffold'] == test_scaffold
    train_mask = ~test_mask
    
    X_train_cv = compute_descriptors(numeric_df.loc[train_mask, 'cleanedMol'].values)
    X_test_cv = compute_descriptors(numeric_df.loc[test_mask, 'cleanedMol'].values)
    
    y_train_cv = numeric_df.loc[train_mask, 'transformed_IC50'].values
    y_test_cv = numeric_df.loc[test_mask, 'transformed_IC50'].values
    
    trained_models, _, _, best_desc, best_model_name, _ = \
        train_and_select(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    
    print(f"Fold - Test on {test_scaffold[:20]}...: {best_model_name} ({best_desc})")
```

---

## Function Reference

### **data_loader.py**
- `load_excel_sheets(file_name)` → list of DataFrames
- `apply_smiles_cleaning(series_dfs)` → list of DataFrames
- `combine_and_deduplicate(series_dfs)` → DataFrame
- `filter_numeric_ic50(df)` → DataFrame
- `standardize_smiles(smiles, verbose=False)` → str or None

### **descriptors.py**
- `rdkit_desc(smiles)` → list
- `ecfp(smiles, radius, nBits)` → numpy array
- `maccs(smiles)` → numpy array
- `mordred_desc(smiles)` → numpy array
- `compute_descriptors(smiles_list)` → dict of numpy arrays

### **clustering.py**
- `run_umap(descriptor_matrix, ...)` → numpy array
- `run_hdbscan(umap_result, min_sizes)` → tuple (clusters, min_size, score)
- `get_scaffold_safe(mol)` → str or None
- `analyze_chemical_space(numeric_df)` → dict
- `plot_chemical_space(numeric_df, ...)` → None

### **model.py**
- `train_and_select(X_train_raw, y_train, X_test_raw, y_test)` → tuple
- `predict_and_antilog(model, X, df_original)` → DataFrame
- `save_predictions(predictions_dict, output_dir)` → None

### **visualization.py**
- `plot_ic50_distribution(data, col)` → None
- `plot_ic50_boxplot(data, col)` → None
- `univariate_analysis(data, col)` → dict
- `plot_umap_clusters(numeric_df)` → None
- `plot_y_true_vs_pred(y_true, y_pred, model_name, desc_name)` → None

---

## Performance Notes

- **UMAP/HDBSCAN**: May take 30+ seconds for first run (numba JIT compilation)
- **Model Training**: ~5-10 minutes for all 84 combinations (6 descriptors × 14 models)
- **Total Pipeline**: ~15-20 minutes for complete run

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'rdkit'" | `pip install rdkit` |
| "UMAP taking too long" | First run has JIT compilation overhead; subsequent runs are fast |
| "Out of memory" | Reduce descriptor types or use smaller train set |
| "Clusters all -1" | Adjust HDBSCAN min_sizes or UMAP parameters |
| "Best R² very low" | Check data quality, consider feature engineering |
