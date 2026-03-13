#!/usr/bin/env python
"""
Simple test to verify the pipeline works step by step.
"""
import sys
import os

print("\n" + "="*70)
print("QSAR PIPELINE - SIMPLE TEST")
print("="*70)

# Test 1: Check file exists
print("\n✓ Test 1: Checking files...")
if os.path.exists("TB Project QSAR.xlsx"):
    size_mb = os.path.getsize("TB Project QSAR.xlsx") / (1024*1024)
    print(f"  ✅ Excel file found ({size_mb:.2f} MB)")
else:
    print("  ❌ Excel file not found!")
    sys.exit(1)

# Test 2: Test imports
print("\n✓ Test 2: Testing imports...")
try:
    import pandas as pd
    print("  ✅ pandas imported")
    import numpy as np
    print("  ✅ numpy imported")
    from rdkit import Chem
    print("  ✅ rdkit imported")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test 3: Load data
print("\n✓ Test 3: Loading Excel file...")
try:
    df_dict = pd.read_excel("TB Project QSAR.xlsx", sheet_name=None)
    print(f"  ✅ Loaded {len(df_dict)} sheets")
    for i, name in enumerate(df_dict.keys()):
        print(f"    Sheet {i}: {name}")
except Exception as e:
    print(f"  ❌ Failed to load Excel: {e}")
    sys.exit(1)

# Test 4: Extract series
print("\n✓ Test 4: Extracting chemical series...")
try:
    series_list = list(df_dict.values())
    if len(series_list) >= 7:
        Series_A = series_list[2].copy()
        Series_B = series_list[3].copy()
        Series_C = series_list[4].copy()
        Series_D = series_list[5].copy()
        Series_E = series_list[6].copy()
        
        print(f"  ✅ Series A: {len(Series_A)} molecules")
        print(f"  ✅ Series B: {len(Series_B)} molecules")
        print(f"  ✅ Series C: {len(Series_C)} molecules")
        print(f"  ✅ Series D: {len(Series_D)} molecules")
        print(f"  ✅ Series E: {len(Series_E)} molecules")
    else:
        print(f"  ❌ Expected at least 7 sheets, got {len(series_list)}")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed to extract series: {e}")
    sys.exit(1)

# Test 5: Check SMILES column
print("\n✓ Test 5: Checking SMILES column...")
try:
    if "Canonical SMILES.1" in Series_A.columns:
        sample_smiles = Series_A["Canonical SMILES.1"].iloc[0]
        print(f"  ✅ SMILES column found: '{sample_smiles}'")
    else:
        print(f"  ❌ SMILES column not found!")
        print(f"     Available columns: {Series_A.columns.tolist()}")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 6: Check IC50 column
print("\n✓ Test 6: Checking IC50 column...")
try:
    if "IC50 uM" in Series_A.columns:
        ic50_values = Series_A["IC50 uM"].dropna()
        print(f"  ✅ IC50 column found with {len(ic50_values)} values")
    else:
        print(f"  ❌ IC50 column not found!")
        print(f"     Available columns: {Series_A.columns.tolist()}")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - Data is ready for pipeline!")
print("="*70)
print("\nNext: Run 'python main.py' to execute the full QSAR pipeline")
print("="*70 + "\n")
