"""
Populate final screening output with QSAR predictions + pharmacophore outputs
for Series C, E, and two additional datasets (A, B).

Run with:
  ./.venv/bin/python reinvent_integration/populate_output_series.py
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from descriptors import ecfp, maccs, mordred_desc, rdkit_desc

INPUT_CSV = ROOT / "pharm_input.csv"
QSAR_MODEL = ROOT / "reinvent_integration" / "artifacts" / "qsar_best_model.joblib"
PHARM_PER_COMPOUND = ROOT / "pharm_outputs" / "screening" / "series_abce_pharm_by_compound.csv"

OUT_DIR = ROOT / "pharm_outputs" / "screening"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "pharm_screen_results.csv"
OUT_CSV_ALT = OUT_DIR / "pharm_screen_results_abce_qsar_pharm.csv"

TARGET_SERIES = {"Series C", "Series E", "Series A", "Series B"}


def _safe_sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def _compute_descriptor_vector(smiles: str, descriptor_name: str) -> np.ndarray:
    if descriptor_name == "RDKit":
        return np.asarray(rdkit_desc(smiles), dtype=float)
    if descriptor_name == "ECFP_r1":
        return np.asarray(ecfp(smiles, radius=1), dtype=float)
    if descriptor_name == "ECFP_r2":
        return np.asarray(ecfp(smiles, radius=2), dtype=float)
    if descriptor_name == "ECFP_r3":
        return np.asarray(ecfp(smiles, radius=3), dtype=float)
    if descriptor_name == "MACCS":
        return np.asarray(maccs(smiles), dtype=float)
    if descriptor_name == "Mordred":
        return np.asarray(mordred_desc(smiles), dtype=float)
    raise ValueError(f"Unsupported descriptor: {descriptor_name}")


def canonicalize(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def predict_pic50(smiles_list, bundle):
    descriptor_name = bundle["descriptor_name"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    rows = []
    valid_mask = []
    for smiles in smiles_list:
        try:
            rows.append(_compute_descriptor_vector(smiles, descriptor_name))
            valid_mask.append(True)
        except Exception:
            rows.append(None)
            valid_mask.append(False)

    valid_rows = [row for row in rows if row is not None]
    if not valid_rows:
        return [0.0 for _ in smiles_list]

    x_matrix = np.vstack(valid_rows)
    x_matrix = imputer.transform(x_matrix)
    if scaler is not None:
        x_matrix = scaler.transform(x_matrix)

    predicted_log_ic50 = model.predict(x_matrix)
    predicted_pic50 = (6.0 - predicted_log_ic50).astype(float)

    output = []
    idx = 0
    for is_valid in valid_mask:
        if is_valid:
            output.append(float(predicted_pic50[idx]))
            idx += 1
        else:
            output.append(0.0)
    return output


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input data: {INPUT_CSV}")
    if not QSAR_MODEL.exists():
        raise FileNotFoundError(f"Missing QSAR model artifact: {QSAR_MODEL}")

    df = pd.read_csv(INPUT_CSV)
    df = df[df["Series"].isin(TARGET_SERIES)].copy()
    df = df.dropna(subset=["Identifier", "Smiles"]).copy()
    df = df[df["Smiles"].astype(str).str.strip() != ""].copy()
    df = df.drop_duplicates(subset=["Identifier"], keep="first").reset_index(drop=True)

    df["canon_smiles"] = df["Smiles"].apply(canonicalize)
    df = df.dropna(subset=["canon_smiles"]).reset_index(drop=True)

    bundle = joblib.load(str(QSAR_MODEL))
    df["pred_pic50"] = predict_pic50(df["canon_smiles"].tolist(), bundle)
    df["qsar_score"] = df["pred_pic50"].apply(lambda x: _safe_sigmoid((x - 6.0) / 1.0))

    # Merge pharmacophore output if available
    if PHARM_PER_COMPOUND.exists():
        pharm = pd.read_csv(PHARM_PER_COMPOUND)
        pharm = pharm.rename(columns={"series": "pharm_series"})
        df = df.merge(pharm, on="Identifier", how="left")
    else:
        df["pharm_y_pred"] = np.nan
        df["pharm_y_score"] = np.nan

    df["pharm_y_pred"] = df["pharm_y_pred"].fillna(0).astype(int)
    df["pharm_y_score"] = df["pharm_y_score"].fillna(0.0).astype(float)

    # Final combined prioritization score
    # weight: QSAR 60%, pharmacophore 40%
    df["combined_score"] = 0.6 * df["qsar_score"] + 0.4 * df["pharm_y_score"].clip(lower=0)

    # Keep useful columns
    out_cols = [
        "Identifier", "Series", "Smiles", "canon_smiles",
        "IC50 uM", "PIC50",
        "pred_pic50", "qsar_score",
        "pharm_y_pred", "pharm_y_score",
        "combined_score",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    out = df[out_cols].sort_values(["Series", "combined_score"], ascending=[True, False]).reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False)
    out.to_csv(OUT_CSV_ALT, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_CSV_ALT}")
    print("Counts by series:")
    print(out["Series"].value_counts().to_string())
    print("Top by combined_score:")
    print(out[["Identifier", "Series", "pred_pic50", "pharm_y_score", "combined_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
