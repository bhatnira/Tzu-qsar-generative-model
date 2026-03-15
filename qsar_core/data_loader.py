"""
data_loader.py
Module for loading and preprocessing QSAR data.
"""
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

def load_excel_sheets(file_name):
    df_dict = pd.read_excel(file_name, sheet_name=None)
    # Extract series by index (adjust if needed)
    series = [list(df_dict.values())[i].copy() for i in range(2, 7)]
    names = ["Series-A-Triazole", "Series-B-Cysteine", "Series-C-Spiro", "Series-D-Pyrrolidine", "Series-E-Spiro"]
    codes = list("ABCDE")
    for df, name, code in zip(series, names, codes):
        df["Series_Name"] = name
        df["Series_Code"] = code
    return series

def validate_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid_SMILES"
        else:
            return True, ""
    except:
        return False, "Parsing_Error"

def preprocess_smiles(smiles):
    if pd.isna(smiles):
        return smiles
    smiles = str(smiles)
    smiles = re.sub(r"\[NH3\]Cl", "N", smiles)
    smiles = re.sub(r"\.\[HCl\]", "", smiles)
    smiles = re.sub(r"(\.\[HCl\])+", "", smiles)
    smiles = smiles.replace("N(=O)(=O)", "[N+](=O)[O-]")
    return smiles

def clean_validate_smiles(smiles, remover=SaltRemover()):
    try:
        smiles = preprocess_smiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid_SMILES", None
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        clean_smiles = Chem.MolToSmiles(mol, canonical=True)
        return True, "", clean_smiles
    except Exception as e:
        return False, str(e), None

def apply_smiles_cleaning(series_dfs):
    for df in series_dfs:
        if "Canonical SMILES.1" in df.columns:
            df[["Valid_SMILES", "SMILES_Error", "Clean_SMILES"]] = df["Canonical SMILES.1"].apply(
                lambda x: pd.Series(clean_validate_smiles(x))
            )
        else:
            print("Column 'Canonical SMILES.1' not found.")
    return series_dfs

def combine_and_deduplicate(series_dfs):
    combined_df = pd.concat(series_dfs, ignore_index=True)
    # Canonicalize
    def canonicalize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
        return None
    combined_df['Canonical_SMILES'] = combined_df['Clean_SMILES'].apply(canonicalize)
    df = combined_df.dropna(subset=['Canonical_SMILES'])
    df = df.drop_duplicates(subset=['Canonical_SMILES'])
    return df

def filter_numeric_ic50(df):
    # Remove non-numeric IC50
    numeric_df = df[pd.to_numeric(df["IC50 uM"], errors="coerce").notna()].copy()
    numeric_df["IC50 uM"] = pd.to_numeric(numeric_df["IC50 uM"], errors="coerce")
    return numeric_df

def standardize_smiles(smiles, verbose=False):
    """Standardize molecule without fixing protonation or nitro groups."""
    from rdkit.Chem.MolStandardize import rdMolStandardize
    
    if verbose: 
        print("Original:", smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if verbose:
            print("Failed to parse SMILES:", smiles)
        return None

    # Cleanup
    clean_mol = rdMolStandardize.Cleanup(mol)
    if verbose:
        print('Remove Hs, disconnect metal atoms, normalize the molecule, reionize the molecule:')

    # Fragment parent
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    if verbose:
        print('Select the "parent" fragment:')

    # Uncharge
    uncharger = rdMolStandardize.Uncharger()
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    if verbose:
        print('Neutralize the molecule:')

    # Tautomer
    te = rdMolStandardize.TautomerEnumerator()
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    if verbose:
        print('Enumerate tautomers:')
    
    if taut_uncharged_parent_clean_mol is None:
        return None

    return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)
