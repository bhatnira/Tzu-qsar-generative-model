import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    from qsar_core.data_loader import load_excel_sheets
    print("✓ data_loader OK")
except Exception as e:
    print("✗ data_loader ERROR:", e)

print("All imports tested!")
