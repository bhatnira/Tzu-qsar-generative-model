"""
Prepare Series C/E SMILES files for REINVENT fine-tuning and similarity guidance.

Outputs:
- reinvent_integration/data/series_ce_smiles.smi
- reinvent_integration/data/series_ce_unique.smi
- reinvent_integration/data/series_ce_labeled.csv

Usage:
    python reinvent_integration/prepare_series_ce_data.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_excel_sheets, apply_smiles_cleaning


def main() -> None:
    excel_path = PROJECT_ROOT / "TB Project QSAR.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing dataset: {excel_path}")

    series_frames = load_excel_sheets(str(excel_path))
    series_frames = apply_smiles_cleaning(series_frames)

    selected_frames = []
    for frame in series_frames:
        code = str(frame.get("Series_Code", pd.Series([""])).iloc[0]) if not frame.empty else ""
        if code in {"C", "E"}:
            copy_frame = frame.copy()
            copy_frame["Series_Label"] = f"Series {code}"
            selected_frames.append(copy_frame)

    if not selected_frames:
        raise RuntimeError("Could not find Series C/E data in the input workbook.")

    combined = pd.concat(selected_frames, ignore_index=True)

    if "Canonical_SMILES" not in combined.columns:
        source_col = None
        for candidate in ["Clean_SMILES", "Canonical SMILES.1", "Canonical SMILES", "SMILES", "Smiles"]:
            if candidate in combined.columns:
                source_col = candidate
                break
        if source_col is None:
            raise RuntimeError("No SMILES column found for Series C/E extraction.")

        def canonicalize(smiles: str):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    return None
                return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                return None

        combined["Canonical_SMILES"] = combined[source_col].apply(canonicalize)

    combined = combined.dropna(subset=["Canonical_SMILES"]).copy()
    combined["Canonical_SMILES"] = combined["Canonical_SMILES"].astype(str).str.strip()
    combined = combined[combined["Canonical_SMILES"] != ""].reset_index(drop=True)

    unique_smiles = sorted(set(combined["Canonical_SMILES"].tolist()))

    out_dir = PROJECT_ROOT / "reinvent_integration" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_smiles_path = out_dir / "series_ce_smiles.smi"
    unique_smiles_path = out_dir / "series_ce_unique.smi"
    labeled_csv_path = out_dir / "series_ce_labeled.csv"

    with all_smiles_path.open("w", encoding="utf-8") as handle:
        for smiles in combined["Canonical_SMILES"].tolist():
            handle.write(f"{smiles}\n")

    with unique_smiles_path.open("w", encoding="utf-8") as handle:
        for smiles in unique_smiles:
            handle.write(f"{smiles}\n")

    combined.to_csv(labeled_csv_path, index=False)

    print(f"Series C/E rows: {len(combined)}")
    print(f"Series C/E unique SMILES: {len(unique_smiles)}")
    print(f"Saved: {all_smiles_path}")
    print(f"Saved: {unique_smiles_path}")
    print(f"Saved: {labeled_csv_path}")


if __name__ == "__main__":
    main()
