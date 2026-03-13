"""
Run pharmacophore 3D alignment screening for selected series compounds
(Series C, E, A, B) and export per-compound hit scores.

Run with:
  $SCHRODINGER/run python3 reinvent_integration/pharm_screen_series.py
"""

import os
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pharmacophore_pipeline as pp
from schrodinger.structure import StructureWriter

INPUT_CSV = ROOT / "pharm_input.csv"
OUT_DIR = ROOT / "pharm_outputs" / "screening"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_MAE = OUT_DIR / "series_abce_raw.mae"
PREP_MAE = OUT_DIR / "series_abce_prepared.maegz"
CONF_MAE = OUT_DIR / "series_abce_conformers.maegz"

BEST_HYPO = ROOT / "pharm_outputs" / "hypotheses" / "pharm_hypo_find_common" / "pharm_hypo_find_common" / "hypotheses" / "AHHR_1.phypo"

JOBNAME = "series_abce_pharm"
SCREEN_RAW_CSV = OUT_DIR / f"{JOBNAME}_results.csv"
SCREEN_BY_COMPOUND_CSV = OUT_DIR / "series_abce_pharm_by_compound.csv"

TARGET_SERIES = {"Series C", "Series E", "Series A", "Series B"}


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")
    if not BEST_HYPO.exists():
        raise FileNotFoundError(f"Missing pharmacophore hypothesis: {BEST_HYPO}")

    df = pd.read_csv(INPUT_CSV)
    df = df[df["Series"].isin(TARGET_SERIES)].copy()
    df = df.dropna(subset=["Identifier", "Smiles"]).copy()
    df = df[df["Smiles"].astype(str).str.strip() != ""].copy()
    df = df.drop_duplicates(subset=["Identifier"], keep="first").reset_index(drop=True)

    print(f"Selected compounds: {len(df)}")
    print(df["Series"].value_counts().to_string())

    with StructureWriter(str(RAW_MAE)) as writer:
        for _, row in df.iterrows():
            title = str(row["Identifier"]).strip()
            smiles = str(row["Smiles"]).strip()
            series = str(row["Series"]).strip()
            activity = 1 if series in {"Series C", "Series E"} else 0
            st = pp.smiles_to_structure(smiles, title=title, activity=activity, series=series)
            if st is not None:
                writer.append(st)

    print(f"Wrote raw structures: {RAW_MAE}")

    pp.NCPUS = min(8, (os.cpu_count() or 8))
    prepared = pp.prepare_ligands(str(RAW_MAE), str(PREP_MAE), ph=7.0, ph_tolerance=2.0)
    conformers = pp.generate_conformers(prepared, str(CONF_MAE), max_confs=20, energy_window=10.0, rmsd_cutoff=1.0)

    metrics = pp.validate_pharmacophore(
        best_phypo=str(BEST_HYPO),
        screen_mae=conformers,
        output_dir=str(OUT_DIR),
        jobname=JOBNAME,
    )
    print(f"Validation metrics: {metrics}")

    if not SCREEN_RAW_CSV.exists():
        raise FileNotFoundError(f"Expected screening results not found: {SCREEN_RAW_CSV}")

    raw = pd.read_csv(SCREEN_RAW_CSV)
    raw["title"] = raw["title"].astype(str)

    # Keep best score per compound title
    raw = raw.sort_values(["title", "y_score"], ascending=[True, False])
    best = raw.groupby("title", as_index=False).first()

    best = best.rename(columns={"title": "Identifier", "y_pred": "pharm_y_pred", "y_score": "pharm_y_score"})
    keep_cols = [c for c in ["Identifier", "series", "pharm_y_pred", "pharm_y_score"] if c in best.columns]
    best = best[keep_cols]
    best.to_csv(SCREEN_BY_COMPOUND_CSV, index=False)

    print(f"Per-compound pharmacophore output: {SCREEN_BY_COMPOUND_CSV}")


if __name__ == "__main__":
    main()
