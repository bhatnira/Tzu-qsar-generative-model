import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    print("Importing pandas...")
    import pandas as pd
    print("✓ pandas OK")
except Exception as e:
    print("✗ pandas ERROR:", e)
    
try:
    print("Importing rdkit...")
    from rdkit import Chem
    print("✓ rdkit OK")
except Exception as e:
    print("✗ rdkit ERROR:", e)

try:
    print("Importing data_loader...")
    from data_loader import load_excel_sheets
    print("✓ data_loader OK")
except Exception as e:
    print("✗ data_loader ERROR:", e)

print("All imports tested!")
