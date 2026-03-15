# QSAR Pipeline Setup Guide

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/bhatnira/Tzu-qsar-generative-model.git
cd Tzu-qsar-generative-model
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Using venv (built-in)
python -m venv qsar_env
source qsar_env/bin/activate  # On macOS/Linux
# or
qsar_env\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r config/requirements.txt
```

## Running the Pipeline

### Main Pipeline Execution
```bash
python scripts/main.py
```

**What it does:**
1. Loads 160 molecules from `TB Project QSAR.xlsx` (5 chemical series)
2. Validates and cleans SMILES strings
3. Standardizes molecular structures using RDKit
4. Computes 6 types of molecular descriptors (8,141 total features)
5. Performs UMAP dimensionality reduction
6. Runs HDBSCAN clustering
7. Trains 84 machine learning model combinations
8. Selects the best model (KNN + RDKit, R² = 0.8868)
9. Generates predictions for Series C and E
10. Saves outputs to `outputs/` folder

### Output Files Generated
- **Prediction CSV files:**
  - `SeriesE_with_predictions.csv` - 6 molecules from Series E
  - `SeriesC_test_with_predictions.csv` - 5 molecules from Series C
  - `Non_numeric_IC50_predictions.csv` - 11 molecules with non-numeric IC50
  - `Dropped_rows_predictions.csv` - 31 molecules with IC50=100 or 200 µM

- **Visualizations:**
  - `ic50_distribution.png` - IC50 value histogram
  - `umap_clusters.png` - Chemical space UMAP projection
  - `y_true_vs_pred.png` - Model prediction accuracy plot

- **Model Results:**
  - `model_results.csv` - R² scores for all 84 model combinations

## Dependencies Breakdown

### Core Data Science
- **numpy** - Numerical computing
- **pandas** - Data manipulation and analysis
- **scipy** - Scientific computing utilities
- **scikit-learn** - Machine learning algorithms (Ridge, Lasso, KNN, RF, etc.)

### Cheminformatics
- **rdkit** - Molecular structure handling, SMILES processing, descriptors
- **mordred** - Advanced molecular descriptors (1,613 descriptors)
- **openpyxl** - Excel file reading (for TB Project QSAR.xlsx)

### Machine Learning Models
- **xgboost** - Gradient boosting machine
- **lightgbm** - Light gradient boosting machine
- **catboost** - Categorical boosting machine

### Dimensionality Reduction & Clustering
- **umap-learn** - UMAP algorithm for dimensionality reduction
- **hdbscan** - Hierarchical density-based clustering

### Visualization
- **matplotlib** - 2D plotting library
- **seaborn** - Statistical visualization

## Project Structure

```
Tzu-qsar-generative-model/
├── main.py                           # Main execution script
├── data_loader.py                    # Data loading & preprocessing
├── descriptors.py                    # Molecular descriptor computation
├── clustering.py                     # UMAP & HDBSCAN clustering
├── model.py                          # ML model training & evaluation
├── visualization.py                  # Plotting functions
├── TB Project QSAR.xlsx              # Input data (160 molecules)
├── outputs/                          # Generated predictions & plots
├── QSAR_Debug_Pipeline.ipynb         # Complete pipeline as Jupyter notebook
├── config/requirements.txt                  # All Python dependencies
└── README.md                         # Project documentation
```

## Troubleshooting

### Issue: Module not found errors
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r config/requirements.txt --upgrade
```

### Issue: RDKit import error
**Solution:** RDKit is best installed via conda:
```bash
conda install -c conda-forge rdkit
```

### Issue: Excel file not found
**Solution:** Ensure `TB Project QSAR.xlsx` is in the root directory:
```bash
ls -la TB\ Project\ QSAR.xlsx
```

### Issue: Memory errors with descriptor computation
**Solution:** The descriptor computation is memory-intensive for large datasets. If you experience memory issues, try reducing the batch size in `descriptors.py` or run on a machine with more RAM.

## Model Performance

**Best Model:** K-Nearest Neighbors (KNN) with RDKit Descriptors
- **R² Score (Test Set):** 0.8868
- **Test Set Size:** 5 molecules (Series C)
- **Training Set Size:** 138 molecules (Series A, B, D)
- **Descriptors Used:** RDKit (217 features)

## File Descriptions

### data_loader.py
Functions for loading Excel sheets, validating SMILES, cleaning molecular structures, deduplicating data, and filtering for numeric IC50 values.

### descriptors.py
Computes 6 types of molecular descriptors:
1. RDKit descriptors (217)
2. ECFP-r1 (2048 bits)
3. ECFP-r2 (2048 bits)
4. ECFP-r3 (2048 bits)
5. MACCS keys (167 bits)
6. Mordred descriptors (1,613)

### clustering.py
Implements UMAP for dimensionality reduction and HDBSCAN for clustering. Also includes Murcko scaffold extraction.

### model.py
Trains and evaluates 14 different regression models:
- Ridge Regression
- Lasso Regression
- Elastic Net
- Bayesian Ridge
- Huber Regressor
- K-Nearest Neighbors
- Support Vector Regressor
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### visualization.py
Creates publication-quality plots for IC50 distribution, UMAP clustering, and model prediction accuracy.

## Next Steps

1. **Analyze Results:** Examine the generated CSV files and visualizations
2. **Hyperparameter Tuning:** Modify model parameters in `model.py` for better performance
3. **Feature Importance:** Investigate which descriptors drive predictions
4. **Validation:** Use the trained model on new molecules with `generate_all_predictions.py`

## Support

For issues or questions, refer to the main `README.md` or check the `QSAR_Debug_Pipeline.ipynb` for step-by-step execution details.
