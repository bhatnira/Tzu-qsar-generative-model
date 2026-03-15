"""
post_screen.py
Post-generation pipeline: QSAR prediction + Pharmacophore 3D alignment screening.

Runs on all three generation mode sample CSVs (similarity, scaffold_hopping,
scaffold_generation) and produces a single ranked report:
  reinvent_integration/results/post_screen_results.csv

Steps:
  1. Load all sample CSVs, keep valid unique SMILES
  2. QSAR pIC50 prediction (fast, RDKit/KNN model)
  3. Filter top-N by predicted pIC50 for pharmacophore screening
  4. Run Schrödinger Phase 3D alignment (LigPrep → ConfGen → phase_screen)
  5. Combine all scores → ranked CSV + summary table
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from qsar_core.descriptors import ecfp, maccs, mordred_desc, rdkit_desc


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_FILES = {
    "similarity":          ROOT / "reinvent_integration/results/samples_similarity_v2.csv",
    "scaffold_hopping":    ROOT / "reinvent_integration/results/samples_scaffold_hopping_v2.csv",
    "scaffold_generation": ROOT / "reinvent_integration/results/samples_scaffold_generation_v2.csv",
}

QSAR_MODEL_PATH  = ROOT / "reinvent_integration/artifacts/qsar_best_model.joblib"
PHYPO_PATH       = ROOT / "pharm_outputs/hypotheses/pharm_hypo_find_common/pharm_hypo_find_common/hypotheses/AHHR_1.phypo"
REFERENCE_SMILES = ROOT / "reinvent_integration/data/series_ce_unique.smi"
TRAINING_SMILES  = ROOT / "reinvent_integration/data/series_ce_unique.smi"
OUT_CSV          = ROOT / "reinvent_integration/results/post_screen_results.csv"
SCHRODINGER_ROOT = Path("/opt/schrodinger/schrodinger2026-1")
MOTIF_SMARTS     = "c1cncc(N2CCC3(CNC3)C2)c1"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def canonicalize(smi: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def _fingerprint(smi: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _compute_descriptor_vector(smiles: str, descriptor_name: str) -> np.ndarray:
    if descriptor_name == "RDKit":
        return np.asarray(rdkit_desc(smiles), dtype=float)
    if descriptor_name.startswith("ECFP_r"):
        r = int(descriptor_name[-1])
        return np.asarray(ecfp(smiles, radius=r), dtype=float)
    if descriptor_name == "MACCS":
        return np.asarray(maccs(smiles), dtype=float)
    if descriptor_name == "Mordred":
        return np.asarray(mordred_desc(smiles), dtype=float)
    raise ValueError(f"Unsupported descriptor: {descriptor_name}")


def predict_pic50(smiles_list: List[str], model_bundle: Dict) -> np.ndarray:
    desc    = model_bundle["descriptor_name"]
    imputer = model_bundle["imputer"]
    scaler  = model_bundle["scaler"]
    model   = model_bundle["model"]

    rows: List[Optional[np.ndarray]] = []
    for smi in smiles_list:
        try:
            rows.append(_compute_descriptor_vector(smi, desc))
        except Exception:
            rows.append(None)

    valid_rows  = [r for r in rows if r is not None]
    valid_mask  = [r is not None for r in rows]
    if not valid_rows:
        return np.zeros(len(smiles_list))

    X = np.vstack(valid_rows)
    X = imputer.transform(X)
    if scaler is not None:
        X = scaler.transform(X)

    pred_log_ic50 = model.predict(X)
    # scorer formula: pIC50 = 6 - log10(IC50_uM)
    pic50_seq = (6.0 - pred_log_ic50).astype(float)

    out = np.zeros(len(smiles_list))
    idx = 0
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            out[i] = pic50_seq[idx]
            idx += 1
    return out


def load_reference_fps(path: Path, radius: int = 2, n_bits: int = 2048):
    fps = []
    if not path.exists():
        return fps
    with path.open() as fh:
        for line in fh:
            tokens = line.strip().split()
            if tokens:
                fp = _fingerprint(tokens[0], radius, n_bits)
                if fp is not None:
                    fps.append(fp)
    return fps


def max_tanimoto(smiles_list: List[str], ref_fps, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    if not ref_fps:
        return np.zeros(len(smiles_list))
    out = []
    for smi in smiles_list:
        fp = _fingerprint(smi, radius, n_bits)
        if fp is None:
            out.append(0.0)
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            out.append(max(sims) if sims else 0.0)
    return np.array(out)


def check_motif(smiles_list: List[str], smarts: str) -> np.ndarray:
    query = Chem.MolFromSmarts(smarts)
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        out.append(1 if (mol and query and mol.HasSubstructMatch(query)) else 0)
    return np.array(out)


# ─────────────────────────────────────────────────────────────────────────────
# Pharmacophore Screening (Schrödinger Phase)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_screening(
    smiles_list: List[str],
    hypothesis_path: Path,
    schrodinger_root: Path,
    confgen_max: int = 50,
    confgen_energy_window: float = 10.0,
) -> pd.DataFrame:
    """
    Run LigPrep → ConfGen → phase_screen for a list of SMILES.
    Returns DataFrame with columns: smiles_idx, fit_score, rmsd, matched_features.
    """

    n = len(smiles_list)
    empty_df = pd.DataFrame({
        "smiles_idx":       range(n),
        "phase_fit_score":  [0.0] * n,
        "phase_rmsd":       [999.0] * n,
        "phase_matched_ft": [0.0] * n,
        "phase_screened":   [False] * n,
    })

    if not hypothesis_path.exists():
        print(f"  [Phase] Hypothesis not found: {hypothesis_path}")
        return empty_df

    phase_exe   = schrodinger_root / "phase_screen"
    ligprep_exe = schrodinger_root / "ligprep"
    confgen_exe = schrodinger_root / "confgenx"

    for exe in [phase_exe, ligprep_exe, confgen_exe]:
        if not exe.exists():
            print(f"  [Phase] Missing executable: {exe}")
            return empty_df

    try:
        from schrodinger.structure import SmilesStructure, StructureReader, StructureWriter
        schrodinger_available = True
    except Exception as exc:
        print(f"  [Phase] Cannot import Schrödinger Python API: {exc}")
        print("  [Phase] Running phase_screen via subprocess SMILES input instead...")
        schrodinger_available = False

    results: Dict[int, Dict] = {}

    with tempfile.TemporaryDirectory(prefix="post_screen_") as tmp_str:
        tmp = Path(tmp_str)
        ligands_smi = tmp / "ligands.smi"
        ligands_mae  = tmp / "ligands.mae"
        prepared_mae = tmp / "prepared.maegz"
        confgen_mae  = tmp / "prepared-out.maegz"
        out_job = "phase_batch"

        # ── Write input ──────────────────────────────────────────────────────
        with ligands_smi.open("w") as fh:
            for idx, smi in enumerate(smiles_list):
                fh.write(f"{smi} mol_{idx}\n")

        # ── LigPrep ──────────────────────────────────────────────────────────
        print(f"  [Phase] LigPrep on {n} molecules...")
        ligprep_cmd = [
            str(ligprep_exe),
            "-ismi", str(ligands_smi),
            "-omae", str(prepared_mae),
            "-ph", "7.0", "-pht", "2.0", "-s", "1",
            "-WAIT", "-NOJOBID",
        ]
        result = subprocess.run(ligprep_cmd, cwd=tmp, capture_output=True, text=True)
        if result.returncode != 0 or not prepared_mae.exists():
            print(f"  [Phase] LigPrep failed: {result.stderr[:500]}")
            return empty_df

        # ── ConfGen ──────────────────────────────────────────────────────────
        print(f"  [Phase] ConfGen (max {confgen_max} confs, ΔE ≤ {confgen_energy_window})...")
        confgen_cmd = [
            str(confgen_exe),
            str(prepared_mae),
            "-m", str(max(1, confgen_max)),
            "-ewindow", str(max(0.1, confgen_energy_window)),
            "-optimize",
            "-force_field", "OPLS_2005",
            "-WAIT", "-NOJOBID",
        ]
        result = subprocess.run(confgen_cmd, cwd=tmp, capture_output=True, text=True)

        # Locate ConfGen output (tries multiple name patterns)
        confgen_out = None
        for pat in ["*-out.maegz", "*-out.mae", "*.maegz", "*.mae"]:
            hits = [p for p in tmp.glob(pat) if "prepared" in p.name and "out" in p.name]
            if hits:
                confgen_out = hits[0]
                break
        if confgen_out is None:
            for pat in ["*out*.maegz", "*out*.mae"]:
                hits = sorted(tmp.glob(pat))
                if hits:
                    confgen_out = hits[0]
                    break
        if confgen_out is None:
            print(f"  [Phase] ConfGen produced no output. stderr: {result.stderr[:300]}")
            return empty_df

        # ── Phase Screen ─────────────────────────────────────────────────────
        print(f"  [Phase] phase_screen against {hypothesis_path.name}...")
        cmd = [
            str(phase_exe),
            str(confgen_out),
            str(hypothesis_path),
            out_job,
            "-distinct",
            "-HOST", "localhost:1",
            "-NJOBS", "1",
        ]
        result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [Phase] phase_screen failed: {result.stderr[:300]}")
            return empty_df

        # ── Parse hits ───────────────────────────────────────────────────────
        hit_file = None
        for name in [
            f"{out_job}_hits.maegz", f"{out_job}_hits.mae",
            f"{out_job}.maegz",     f"{out_job}.mae",
            f"{out_job}-out.maegz", f"{out_job}-out.mae",
        ]:
            candidate = tmp / name
            if candidate.exists():
                hit_file = candidate
                break
        if hit_file is None:
            hits = sorted(tmp.glob("*hits*"))
            if hits:
                hit_file = hits[0]

        if hit_file is None:
            print("  [Phase] No hit file found — 0 pharmacophore matches.")
            return empty_df

        print(f"  [Phase] Parsing hits from {hit_file.name}...")
        try:
            from schrodinger.structure import StructureReader
            for st in StructureReader(str(hit_file)):
                title = st.title or ""
                m = re.match(r"^mol_(\d+)", title)
                if m is None:
                    continue
                idx = int(m.group(1))
                if idx < 0 or idx >= n:
                    continue
                fit_score = 0.0
                rmsd = 999.0
                matched_ft = 0.0
                try:
                    for key in ["r_phase_ScreenScore", "r_phase_Score", "r_phase_Fitness"]:
                        v = st.property.get(key)
                        if v is not None:
                            fit_score = float(v)
                            break
                    for key in ["r_phase_RMSD", "r_phase_AlignRMSD"]:
                        v = st.property.get(key)
                        if v is not None:
                            rmsd = float(v)
                            break
                    for key in ["r_i_phase_MatchedFeatures", "i_phase_MatchedFeatures"]:
                        v = st.property.get(key)
                        if v is not None:
                            matched_ft = float(v)
                            break
                except Exception:
                    pass
                if idx not in results or fit_score > results[idx]["phase_fit_score"]:
                    results[idx] = {
                        "phase_fit_score":  fit_score,
                        "phase_rmsd":       rmsd,
                        "phase_matched_ft": matched_ft,
                        "phase_screened":   True,
                    }
        except Exception as exc:
            print(f"  [Phase] Error parsing hit file: {exc}")

    # Build output DataFrame
    rows = []
    for i in range(n):
        if i in results:
            rows.append({"smiles_idx": i, **results[i]})
        else:
            rows.append({
                "smiles_idx":       i,
                "phase_fit_score":  0.0,
                "phase_rmsd":       999.0,
                "phase_matched_ft": 0.0,
                "phase_screened":   True,  # was submitted, just no hit
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n-pharm", type=int, default=50,
                    help="Top-N compounds by predicted pIC50 to submit for pharmacophore screening")
    ap.add_argument("--confgen-max", type=int, default=50)
    ap.add_argument("--confgen-energy-window", type=float, default=10.0)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load sample CSVs ──────────────────────────────────────────────────
    print("[1/5] Loading generated sample CSVs...")
    frames = []
    for mode, csv_path in SAMPLE_FILES.items():
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — skipping {mode}")
            continue
        df = pd.read_csv(csv_path)
        df["mode"] = mode
        frames.append(df)
    if not frames:
        print("ERROR: No sample CSVs found. Run staged learning + sampling first.")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(all_df)}")

    # Keep valid SMILES (SMILES_state == 1)
    if "SMILES_state" in all_df.columns:
        all_df = all_df[all_df["SMILES_state"] == 1].copy()
    elif "Valid" in all_df.columns:
        all_df = all_df[all_df["Valid"] == True].copy()

    print(f"  Valid rows: {len(all_df)}")

    # Canonicalize
    all_df["canon_smiles"] = all_df["SMILES"].apply(canonicalize)
    all_df = all_df.dropna(subset=["canon_smiles"])
    all_df = all_df.drop_duplicates(subset=["canon_smiles"]).reset_index(drop=True)
    print(f"  Unique valid: {len(all_df)}")

    smiles_list = all_df["canon_smiles"].tolist()

    # ── 2. Load reference SMILES (training set novelty check) ────────────────
    print("[2/5] Loading reference SMILES for novelty + similarity...")
    ref_fps = load_reference_fps(REFERENCE_SMILES)
    training_smis = set()
    if TRAINING_SMILES.exists():
        with TRAINING_SMILES.open() as fh:
            for line in fh:
                tokens = line.strip().split()
                if tokens:
                    c = canonicalize(tokens[0])
                    if c:
                        training_smis.add(c)

    all_df["novel"] = all_df["canon_smiles"].apply(lambda s: s not in training_smis)
    all_df["max_tanimoto_ref"] = max_tanimoto(smiles_list, ref_fps)
    all_df["has_motif"] = check_motif(smiles_list, MOTIF_SMARTS)

    # ── 3. QSAR pIC50 prediction ─────────────────────────────────────────────
    print("[3/5] Predicting pIC50 with QSAR model...")
    if not QSAR_MODEL_PATH.exists():
        print("  WARNING: QSAR model not found. Run build_qsar_artifact.py first.")
        all_df["pred_pic50"] = 0.0
    else:
        bundle = joblib.load(str(QSAR_MODEL_PATH))
        print(f"  Model: {type(bundle['model']).__name__} | Descriptor: {bundle['descriptor_name']}")
        all_df["pred_pic50"] = predict_pic50(smiles_list, bundle)
        print(f"  pIC50 range: {all_df['pred_pic50'].min():.2f} – {all_df['pred_pic50'].max():.2f}")
        print(f"  Compounds with pred_pIC50 ≥ 6: {(all_df['pred_pic50'] >= 6.0).sum()}")

    # ── 4. Pharmacophore 3D alignment screening ──────────────────────────────
    print(f"[4/5] Pharmacophore 3D alignment screening (top {args.top_n_pharm} by pIC50)...")

    # Sort by pIC50, take top-N novel, unique
    novel_df = all_df[all_df["novel"] == True].copy()
    top_screen = novel_df.nlargest(args.top_n_pharm, "pred_pic50").reset_index(drop=True)
    top_smiles = top_screen["canon_smiles"].tolist()

    print(f"  Submitting {len(top_smiles)} compounds for Phase screening...")
    phase_df = run_phase_screening(
        smiles_list=top_smiles,
        hypothesis_path=PHYPO_PATH,
        schrodinger_root=SCHRODINGER_ROOT,
        confgen_max=args.confgen_max,
        confgen_energy_window=args.confgen_energy_window,
    )

    # Merge phase results back onto top_screen
    top_screen["phase_fit_score"]  = phase_df["phase_fit_score"].values
    top_screen["phase_rmsd"]       = phase_df["phase_rmsd"].values
    top_screen["phase_matched_ft"] = phase_df["phase_matched_ft"].values
    top_screen["phase_screened"]   = True

    # Fill non-screened with NaN
    screened_canon = set(top_screen["canon_smiles"])
    all_df["phase_fit_score"]  = np.nan
    all_df["phase_rmsd"]       = np.nan
    all_df["phase_matched_ft"] = np.nan
    all_df["phase_screened"]   = False

    for _, row in top_screen.iterrows():
        mask = all_df["canon_smiles"] == row["canon_smiles"]
        all_df.loc[mask, "phase_fit_score"]  = row["phase_fit_score"]
        all_df.loc[mask, "phase_rmsd"]       = row["phase_rmsd"]
        all_df.loc[mask, "phase_matched_ft"] = row["phase_matched_ft"]
        all_df.loc[mask, "phase_screened"]   = True

    pharm_hits = all_df[all_df["phase_fit_score"] > 0].shape[0]
    print(f"  Pharmacophore hits (fit > 0): {pharm_hits}")

    # ── 5. Final scoring & ranking ───────────────────────────────────────────
    print("[5/5] Computing combined score and saving results...")

    # Combined score = 0.50 * sigmoid(pIC50 - 6) + 0.30 * phase_fit + 0.20 * sim
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    all_df["qsar_score"] = sigmoid(all_df["pred_pic50"] - 6.0)
    all_df["phase_score_norm"] = all_df["phase_fit_score"].fillna(0.0).clip(0, 3) / 3.0
    all_df["combined_score"] = (
        0.50 * all_df["qsar_score"]
        + 0.30 * all_df["phase_score_norm"]
        + 0.20 * all_df["max_tanimoto_ref"]
    )

    # Sort by combined score (novel, screened first)
    result = all_df.sort_values(
        ["phase_screened", "combined_score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    out_cols = [
        "canon_smiles", "mode", "pred_pic50", "qsar_score",
        "phase_fit_score", "phase_rmsd", "phase_matched_ft",
        "phase_screened", "max_tanimoto_ref", "has_motif", "novel",
        "combined_score",
    ]
    if "NLL" in result.columns:
        out_cols.append("NLL")

    out_cols = [c for c in out_cols if c in result.columns]
    result[out_cols].to_csv(out_path, index=False)
    print(f"\n✓ Results saved → {out_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(" POST-GENERATION SCREENING SUMMARY")
    print("═" * 65)
    for mode in all_df["mode"].unique():
        sub = all_df[all_df["mode"] == mode]
        nov = sub[sub["novel"] == True]
        ph  = sub[sub["phase_fit_score"] > 0]
        mot = sub[sub["has_motif"] == 1]
        print(f"\n  Mode: {mode}")
        print(f"    Unique valid       : {len(sub)}")
        print(f"    Novel              : {len(nov)}")
        print(f"    Has motif          : {mot['has_motif'].sum()}")
        print(f"    Pharm hits (fit>0) : {len(ph)}")
        if len(nov) > 0:
            print(f"    Mean pred_pIC50    : {nov['pred_pic50'].mean():.3f}")
            print(f"    Max  pred_pIC50    : {nov['pred_pic50'].max():.3f}")

    print("\n  Top 10 overall (by combined score):")
    top10 = result[result["novel"] == True].head(10)
    pd.set_option("display.max_colwidth", 40)
    print(top10[["canon_smiles", "mode", "pred_pic50", "phase_fit_score",
                  "max_tanimoto_ref", "has_motif", "combined_score"]].to_string(index=False))
    print("═" * 65)


if __name__ == "__main__":
    main()
