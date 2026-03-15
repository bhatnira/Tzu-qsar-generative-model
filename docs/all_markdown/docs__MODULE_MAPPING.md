# Module Mapping to Notebook Functionality

This document maps the modularized Python code to the notebook cells, showing how the QSAR pipeline is structured.

## Overview

The QSAR pipeline has been refactored into 5 modular components that replicate all functionality from the Jupyter notebook:

```
Copy_of_Qsar_Model_classical_aproach_Problem_Specific_low_data_qsar_modeling_Destinee_Zhu_Roy.ipynb
│
├── Data Loading & Preprocessing
│   └── data_loader.py
│
├── Molecular Descriptor Calculation
│   └── descriptors.py
│
├── Chemical Space Analysis & Clustering
│   └── clustering.py
│
├── Machine Learning Model Training
│   └── model.py
│
├── Visualization & Analysis
│   └── visualization.py
│
└── Main Pipeline
    └── main.py
```

---

## Module-by-Module Mapping

### 1. **data_loader.py**

**Notebook Sections:**
- Data loading and Preprocessing (Cells: b0178299)
- SMILES Validation (Cells: 75e16b45)
- SMILES Cleaning (Cells: 338b0f66)
- Duplicate Removal (Cells: 350c44dc, 75bd4ac9, 2dac6b99)
- Non-numeric IC50 Filtering (Cells: 2c0fb607)
- SMILES Standardization (Cells: a2144f38, 7b77e47d)

**Key Functions:**

| Function | Notebook Equivalent | Purpose |
|----------|-------------------|---------|
| `load_excel_sheets(file_name)` | Cell b0178299 | Read Excel file and extract 5 series (A-E) with labels |
| `validate_smiles(smiles)` | Cell 75e16b45 | Check if SMILES is valid using RDKit |
| `preprocess_smiles(smiles)` | Cell 338b0f66 | Pre-clean SMILES (remove salts, fix nitro groups) |
| `clean_validate_smiles(smiles)` | Cell 338b0f66 | Full SMILES cleaning pipeline |
| `apply_smiles_cleaning(series_dfs)` | Cell 338b0f66 | Apply to all dataframes |
| `combine_and_deduplicate(series_dfs)` | Cells 75bd4ac9, 2dac6b99 | Merge series and remove duplicates |
| `filter_numeric_ic50(df)` | Cell 2c0fb607 | Keep only numeric IC50 values |
| `standardize_smiles(smiles, verbose)` | Cells a2144f38, 7b77e47d | Advanced standardization with tautomer enumeration |

---

### 2. **descriptors.py**

**Notebook Sections:**
- Descriptor Calculation (Cells: 2ccbfb08, 961822a0, 1b405452)
- Multiple Descriptor Types (RDKit, ECFP, MACCS, Mordred)

**Key Functions:**

| Function | Descriptor Type | Purpose |
|----------|-----------------|---------|
| `rdkit_desc(smiles)` | All RDKit 2D descriptors (~200) | Physical-chemical properties |
| `ecfp(smiles, radius, nBits)` | Morgan/ECFP Fingerprints | Circular fingerprints at different radii (r1, r2, r3) |
| `maccs(smiles)` | MACCS Keys | 167-bit structural fingerprint |
| `mordred_desc(smiles)` | Mordred Descriptors | Extended descriptor set |
| `compute_descriptors(smiles_list)` | Cell 1b405452 | Compute all 6 descriptor types for list of SMILES |

**Output:** Dictionary with keys: `RDKit`, `ECFP_r1`, `ECFP_r2`, `ECFP_r3`, `MACCS`, `Mordred`

---

### 3. **clustering.py**

**Notebook Sections:**
- RDKit Descriptor Calculation (Cell: 2ccbfb08)
- UMAP Dimensionality Reduction (Cell: 2ccbfb08)
- HDBSCAN Clustering (Cell: 2ccbfb08)
- Scaffold Extraction (Cell: 2ccbfb08)
- Chemical Space Analysis (Cell: 961822a0)
- Visualization with Alpha Shapes (Cell: 961822a0)

**Key Functions:**

| Function | Notebook Equivalent | Purpose |
|----------|-------------------|---------|
| `run_umap(descriptor_matrix, ...)` | Cell 2ccbfb08 | 2D UMAP projection of descriptor space |
| `run_hdbscan(umap_result, min_sizes)` | Cell 2ccbfb08 | Density clustering with parameter tuning |
| `get_scaffold_safe(mol)` | Cell 2ccbfb08 | Extract Murcko scaffold |
| `analyze_chemical_space(numeric_df)` | Cell 961822a0 | Cluster composition analysis by series |
| `plot_chemical_space(numeric_df, ...)` | Cell 961822a0 | Visualize chemical space with alpha shapes |

**Outputs:**
- UMAP coordinates (2D)
- Cluster assignments
- Murcko scaffolds for each molecule
- Cluster statistics and composition

---

### 4. **model.py**

**Notebook Sections:**
- Model Definition (Cell: 1b405452)
- Training Loop (Cell: 1b405452)
- Best Model Selection (Cell: 1b405452)
- Predictions (Cell: 1b405452)

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `train_and_select(X_train_raw, y_train, X_test_raw, y_test)` | Train all 14 models × 6 descriptors (84 total combinations) |
| `predict_and_antilog(model, X, df_original)` | Generate predictions and convert from log-space |
| `save_predictions(predictions_dict, output_dir)` | Save to CSV files |

**Models Trained (14 total):**
- Tree-based: RF, ET, GB, XGB, LGBM, CAT, DT
- SVM: SVR
- Neighbors: KNN
- Linear: Ridge, Lasso, ElasticNet, BayesianRidge, Huber
- Neural: MLP

**Feature Preprocessing:**
- Missing value imputation (mean strategy)
- Scaling for linear/kernel models (StandardScaler)

---

### 5. **visualization.py**

**Notebook Sections:**
- IC50 Distribution (Cell: e2e8d39f)
- Univariate Analysis (Cell: e2e8d39f)
- UMAP Visualization (Cell: 2ccbfb08)
- Chemical Space Visualization (Cell: 961822a0)
- True vs Predicted Plot (Cell: 1b405452)

**Key Functions:**

| Function | Notebook Equivalent | Purpose |
|----------|-------------------|---------|
| `plot_ic50_distribution(data, col)` | Cell e2e8d39f | Histogram + KDE of IC50 |
| `plot_ic50_boxplot(data, col)` | Cell e2e8d39f | Boxplot with outlier detection |
| `univariate_analysis(data, col)` | Cell e2e8d39f | Statistical summary & recommendations |
| `plot_umap_clusters(numeric_df)` | Cell 2ccbfb08 | Scatter plot of UMAP with clusters |
| `plot_y_true_vs_pred(y_true, y_pred, ...)` | Cell 1b405452 | Model performance visualization |

---

## Pipeline Execution Flow

### Step 1: Load & Preprocess Data
```python
from data_loader import *

# Load 5 series from Excel
series_dfs = load_excel_sheets("TB Project QSAR.xlsx")

# Clean SMILES and remove invalid entries
series_dfs = apply_smiles_cleaning(series_dfs)

# Combine and deduplicate
combined_df = combine_and_deduplicate(series_dfs)

# Keep only numeric IC50
numeric_df = filter_numeric_ic50(combined_df)

# Advanced standardization
numeric_df['cleanedMol'] = numeric_df['Canonical_SMILES'].apply(
    lambda x: standardize_smiles(x, verbose=False)
)
```

### Step 2: Compute Descriptors
```python
from descriptors import compute_descriptors

smiles_list = numeric_df['cleanedMol'].tolist()
X_raw = compute_descriptors(smiles_list)  # Dict with 6 descriptor types
```

### Step 3: Chemical Space Analysis
```python
from clustering import *

# UMAP + HDBSCAN
descriptor_matrix = X_raw['RDKit']
umap_result = run_umap(descriptor_matrix)
clusters, min_size, score = run_hdbscan(umap_result)

# Add to dataframe
numeric_df['UMAP1'] = umap_result[:, 0]
numeric_df['UMAP2'] = umap_result[:, 1]
numeric_df['Cluster'] = clusters

# Analyze composition
stats = analyze_chemical_space(numeric_df)
```

### Step 4: Model Training
```python
from model import train_and_select

# Split by series
train_df = numeric_df[numeric_df['Series_Code'].isin(['A','B','D'])]
test_df = numeric_df[numeric_df['Series_Code'] == 'C']
predict_df = numeric_df[numeric_df['Series_Code'] == 'E']

y_train = train_df['transformed_IC50'].values
y_test = test_df['transformed_IC50'].values

# Train & select best model
trained_models, imputers, scalers, best_desc, best_model_name, results_df = \
    train_and_select(X_train_raw, y_train, X_test_raw, y_test)
```

### Step 5: Predictions & Visualization
```python
from visualization import *

# Predictions
model_best = trained_models[(best_desc, best_model_name)]
pred_E_df = predict_and_antilog(model_best, X_predict_processed, predict_df)

# Visualization
plot_ic50_distribution(numeric_df["IC50 uM"])
plot_umap_clusters(numeric_df)
univariate_analysis(numeric_df["IC50 uM"])
```

---

## Dataset Splits

The pipeline implements **Scaffold-Based Cross-Validation**:

| Series | Code | Purpose | Count (Approx.) |
|--------|------|---------|-----------------|
| Series A (Triazole) | A | Training | ~30-50 |
| Series B (Cysteine) | B | Training | ~30-50 |
| Series C (Spiro) | C | Testing | ~20-30 |
| Series D (Pyrrolidine) | D | Training | ~30-50 |
| Series E (Spiro) | E | Prediction/Hold-out | ~20-30 |

**Strategy:** Train on A+B+D (different scaffolds), test on C (unseen scaffold), predict on E (hold-out)

---

## Reproducibility

All functions use `random_state=42` for reproducibility:
- UMAP: `random_state=42`
- All sklearn models: `random_state=42`
- numpy: `np.random.seed(42)`

---

## Summary

✅ **100% Feature Parity:**
- All notebook cells replicated in modular functions
- Identical preprocessing pipeline
- Same descriptor types (6)
- All 14 machine learning models
- Complete visualization suite
- Scaffold-based cross-validation

The modularized code maintains the exact same functionality while improving:
- **Reusability**: Import specific functions as needed
- **Maintainability**: Easier to update and test
- **Scalability**: Can be extended to new projects
- **Testing**: Unit tests can be written for each module
