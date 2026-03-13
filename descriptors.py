"""
descriptors.py
Module for calculating molecular descriptors.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from mordred import Calculator, descriptors as mordred_descriptors
import pandas as pd

def rdkit_desc(smiles):
    """Calculate all RDKit descriptors for a SMILES string."""
    m = Chem.MolFromSmiles(smiles)
    return [f(m) if m else np.nan for _, f in Descriptors._descList]

def ecfp(smiles, radius=2, nBits=2048):
    """Calculate Extended Connectivity Fingerprint (ECFP/Morgan Fingerprint)."""
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits)
    arr = np.zeros(nBits)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr

def maccs(smiles):
    """Calculate MACCS keys (166 bit fingerprint)."""
    m = Chem.MolFromSmiles(smiles)
    return np.array(MACCSkeys.GenMACCSKeys(m)) if m else np.zeros(167)

# Initialize Mordred calculator
calc = Calculator(mordred_descriptors, ignore_3D=True)

def mordred_desc(smiles):
    """Calculate Mordred descriptors."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(len(calc.descriptors))
    result = calc(m)
    # Convert to numpy array, handling any non-numeric values
    arr = np.array([float(x) if not pd.isna(x) else 0.0 for x in result])
    return arr

def compute_descriptors(smiles_list):
    """
    Compute multiple types of descriptors for a list of SMILES.
    
    Returns:
        dict: Dictionary with keys for each descriptor type and values as numpy arrays
    """
    return {
        "RDKit": np.array([rdkit_desc(s) for s in smiles_list]),
        "ECFP_r1": np.array([ecfp(s, 1) for s in smiles_list]),
        "ECFP_r2": np.array([ecfp(s, 2) for s in smiles_list]),
        "ECFP_r3": np.array([ecfp(s, 3) for s in smiles_list]),
        "MACCS": np.array([maccs(s) for s in smiles_list]),
        "Mordred": pd.DataFrame([mordred_desc(s) for s in smiles_list]).fillna(0).values
    }
