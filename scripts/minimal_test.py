#!/usr/bin/env python
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_core.paths import DATASET_XLSX

print("=" * 80)
print("MINIMAL TEST")
print("=" * 80)

# Step 1: Load Excel
print("\n[1] Loading Excel file...")
try:
    import pandas as pd
    xls = pd.ExcelFile(DATASET_XLSX)
    print(f"  Sheet names: {xls.sheet_names}")
    
    sheet = xls.parse(xls.sheet_names[1])
    print(f"  Shape of {xls.sheet_names[1]}: {sheet.shape}")
    print(f"  Columns: {list(sheet.columns)[:5]}")
    print("  ✅ Excel load OK")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 2: Import modules
print("\n[2] Importing modules...")
try:
    from qsar_core.data_loader import load_excel_sheets
    print("  ✓ data_loader imported")
    
    from qsar_core.descriptors import compute_descriptors
    print("  ✓ descriptors imported")
    
    from qsar_core.clustering import run_umap
    print("  ✓ clustering imported")
    
    from qsar_core.model import train_and_select
    print("  ✓ model imported")
    
    from qsar_core.visualization import plot_ic50_distribution
    print("  ✓ visualization imported")
    
    print("  ✅ All modules OK")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
