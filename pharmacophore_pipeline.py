"""
pharmacophore_pipeline.py
=========================
Ligand-based pharmacophore modeling pipeline using the Schrödinger Python API.

Run with:
    $SCHRODINGER/run python pharmacophore_pipeline.py

Workflow:
    1. Balance dataset (actives from Series E + C, inactives from non_numeric)
    2. Convert SMILES → Schrödinger structures
    3. Prepare ligands (LigPrep: H addition, ionization, 3D + OPLS optimization)
    4. Generate conformers (ConfGen: max 100, 10 kcal/mol window, 1.0 Å RMSD)
    5. Build Common Pharmacophore Hypotheses (Phase, 4–6 sites, survival score ranking)
    6. Screen balanced dataset against best hypothesis
    7. Compute ROC-AUC, enrichment factor, confusion matrix
    8. Save all outputs to disk
"""

import os
import sys
import subprocess
import tempfile
import shutil
import random
import logging
import pandas as pd
import numpy as np

# ── Schrödinger core ──────────────────────────────────────────────────────────
from schrodinger import structure
from schrodinger.structure import SmilesStructure, StructureReader, StructureWriter
from schrodinger.structutils import minimize as schrminimize

# ── scikit-learn metrics ──────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve, auc
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SCHRODINGER  = os.environ.get("SCHRODINGER", "/opt/schrodinger/schrodinger2026-1")
RANDOM_SEED  = 42
MAX_CONFS    = 100          # max conformers per ligand
ENERGY_WIN   = 10.0         # kcal/mol window for ConfGen
RMSD_CUTOFF  = 1.0          # Å RMSD pruning threshold
MIN_SITES    = 4            # pharmacophore min sites
MAX_SITES    = 6            # pharmacophore max sites
OUTPUT_DIR   = "pharm_outputs"

# Phase feature type tags (standard mnemonic codes)
FEATURE_TYPES = ["D", "A", "H", "R", "P", "N"]
# D=HBD, A=HBA, H=Hydrophobic, R=Aromatic, P=PositiveIonizable, N=NegativeIonizable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a Schrödinger CLI command and wait for completion
# ─────────────────────────────────────────────────────────────────────────────
def _run_schr(cmd, desc=""):
    """Run a Schrödinger shell command. Raises on non-zero exit."""
    full_cmd = [os.path.join(SCHRODINGER, cmd[0])] + cmd[1:]
    log.info("Running: %s  [%s]", " ".join(full_cmd), desc)
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "SCHRODINGER": SCHRODINGER},
    )
    if result.returncode != 0:
        log.error("STDOUT:\n%s", result.stdout[-2000:])
        log.error("STDERR:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"{desc} failed (exit {result.returncode})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset balancing
# ─────────────────────────────────────────────────────────────────────────────
def balance_dataset(df: pd.DataFrame,
                    smiles_col: str = "Smiles",
                    series_col: str = "Series",
                    seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Split df into actives (Series E + C) and inactives (non_numeric),
    then downsample inactives so both sets have equal size.

    Parameters
    ----------
    df          : DataFrame with `smiles_col` and `series_col`.
    smiles_col  : Column holding SMILES strings.
    series_col  : Column holding series labels.
    seed        : Random state for reproducible sampling.

    Returns
    -------
    Balanced DataFrame with added `activity` (1/0) and `activity_class`
    (active/inactive) columns.
    """
    random.seed(seed)
    np.random.seed(seed)

    actives   = df[df[series_col].isin(["Series E", "Series C"])].copy()
    inactives = df[df[series_col] == "non_numeric"].copy()

    n_active   = len(actives)
    n_inactive = len(inactives)
    log.info("Actives: %d  |  Available inactives: %d", n_active, n_inactive)

    if n_inactive == 0:
        raise ValueError("No inactive compounds found (Series == 'non_numeric').")

    # Sample inactives to match actives count
    n_sample = min(n_active, n_inactive)
    inactives = inactives.sample(n=n_sample, random_state=seed)

    if n_inactive < n_active:
        log.warning(
            "Fewer inactives (%d) than actives (%d). Using all inactives.",
            n_inactive, n_active,
        )

    actives  ["activity"]       = 1
    actives  ["activity_class"] = "active"
    inactives["activity"]       = 0
    inactives["activity_class"] = "inactive"

    balanced = pd.concat([actives, inactives], ignore_index=True)
    balanced = balanced.dropna(subset=[smiles_col]).reset_index(drop=True)
    balanced = balanced[balanced[smiles_col].str.strip() != ""]

    log.info(
        "Balanced dataset: %d actives + %d inactives = %d total",
        balanced["activity"].sum(),
        (balanced["activity"] == 0).sum(),
        len(balanced),
    )
    return balanced


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SMILES → Schrödinger structure
# ─────────────────────────────────────────────────────────────────────────────
def smiles_to_structure(smiles: str,
                         title: str = "",
                         activity: int = None,
                         series: str = "") -> structure.Structure:
    """
    Convert a SMILES string to a Schrödinger Structure object.
    Attaches activity and series as structure properties.

    Parameters
    ----------
    smiles   : Valid SMILES string.
    title    : Structure title / compound name.
    activity : 1 for active, 0 for inactive (stored as property).
    series   : Series label string.

    Returns
    -------
    schrodinger.structure.Structure or None on failure.
    """
    try:
        st = SmilesStructure(smiles)
        # SmilesStructure returns an object with .get2dStructure() / .get3dStructure()
        mol = st.get2dStructure()
        mol.title = title or smiles[:50]

        # Store metadata as structure properties
        if activity is not None:
            mol.property["i_user_activity"] = int(activity)
        if series:
            mol.property["s_user_series"] = str(series)
        mol.property["s_user_smiles"] = smiles

        return mol

    except Exception as exc:
        log.warning("SMILES conversion failed for '%s': %s", smiles, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ligand preparation (LigPrep)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_ligands(input_mae: str,
                    output_mae: str,
                    ph: float = 7.0,
                    ph_tolerance: float = 2.0) -> str:
    """
    Run LigPrep on an input .mae file:
      - Add hydrogens
      - Generate ionization states at given pH
      - Generate 3D coordinates
      - OPLS4 geometry optimization

    Parameters
    ----------
    input_mae  : Path to input .mae file.
    output_mae : Path for prepared output .mae file.
    ph         : Target pH for ionization.
    ph_tolerance : ±tolerance around target pH.

    Returns
    -------
    Path to the prepared .mae file.
    """
    log.info("Running LigPrep: %s → %s", input_mae, output_mae)

    cmd = [
        "ligprep",
        "-imae",  input_mae,
        "-omae",  output_mae,
        "-ph",    str(ph),
        "-pht",   str(ph_tolerance),
        "-s",     "1",           # 1 ionization state per compound
        "-WAIT",                 # block until complete
        "-NOJOBID",              # no job server needed for single-machine use
    ]
    _run_schr(cmd, desc="LigPrep")
    log.info("LigPrep complete: %s", output_mae)
    return output_mae


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Conformer generation (ConfGen)
# ─────────────────────────────────────────────────────────────────────────────
def generate_conformers(input_mae: str,
                        output_mae: str,
                        max_confs: int = MAX_CONFS,
                        energy_window: float = ENERGY_WIN,
                        rmsd_cutoff: float = RMSD_CUTOFF) -> str:
    """
    Run ConfGen on a LigPrep-prepared .mae file to generate multiple conformers.

    Parameters
    ----------
    input_mae    : Prepared ligands .mae.
    output_mae   : Output conformer .mae.
    max_confs    : Maximum conformers per compound.
    energy_window: Energy window above minimum (kcal/mol).
    rmsd_cutoff  : RMSD pruning threshold (Å).

    Returns
    -------
    Path to the conformer .mae file.
    """
    log.info("Running ConfGen: %s → %s", input_mae, output_mae)

    # Write ConfGen input file (.inp)
    inp_file = output_mae.replace(".mae", ".inp").replace(".maegz", ".inp")
    inp_content = f"""\
INPUT_FILE   {os.path.abspath(input_mae)}
OUTPUT_FILE  {os.path.abspath(output_mae)}
MAX_CONFS    {max_confs}
ENERGY_WINDOW {energy_window}
RMSD_CUTOFF  {rmsd_cutoff}
OPLS         OPLS_2005
AMIDE_MODE   penal
"""
    with open(inp_file, "w") as fh:
        fh.write(inp_content)

    cmd = ["confgen", inp_file, "-WAIT", "-NOJOBID"]
    _run_schr(cmd, desc="ConfGen")
    log.info("ConfGen complete: %s", output_mae)
    return output_mae


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build pharmacophore hypotheses (Phase CPH)
# ─────────────────────────────────────────────────────────────────────────────
def build_pharmacophore(actives_mae: str,
                        output_dir: str,
                        jobname: str = "pharm_hypo",
                        min_sites: int = MIN_SITES,
                        max_sites: int = MAX_SITES,
                        features: list = None) -> str:
    """
    Run Phase Common Pharmacophore Hypothesis (CPH) generation on ACTIVE compounds.

    Steps:
      a) Identify Phase pharmacophore features for each conformer.
      b) Find common feature patterns across actives.
      c) Score and rank hypotheses by Phase survival score.

    Parameters
    ----------
    actives_mae : Conformer .mae containing ONLY active compounds.
    output_dir  : Directory to write hypothesis files.
    jobname     : Base name for all Phase output files.
    min_sites   : Minimum number of pharmacophore sites.
    max_sites   : Maximum number of pharmacophore sites.
    features    : List of feature type codes (default: D,A,H,R,P,N).

    Returns
    -------
    Path to the best hypothesis .phypo file.
    """
    os.makedirs(output_dir, exist_ok=True)
    features = features or FEATURE_TYPES

    log.info("Building pharmacophore hypotheses from: %s", actives_mae)
    log.info("Feature types: %s  |  Sites: %d–%d", ", ".join(features), min_sites, max_sites)

    # ── Step A: Write Phase input file (.inp) ────────────────────────────────
    inp_file  = os.path.join(output_dir, f"{jobname}.inp")
    out_prefix = os.path.join(output_dir, jobname)

    inp_lines = [
        f"JOBNAME  {jobname}",
        f"INPUT_FILE_COLUMN  {os.path.abspath(actives_mae)}",
        f"FEATURES  {' '.join(features)}",
        f"MIN_SITES  {min_sites}",
        f"MAX_SITES  {max_sites}",
        "PFEAT_SCORE_CUTOFF  0.5",
        "MAX_SKIPPED_MOLS  0",
        "SCORE_CUTOFF  0.5",     # survival score threshold
        "ACTIVITY_COLUMN  i_user_activity",
    ]
    with open(inp_file, "w") as fh:
        fh.write("\n".join(inp_lines) + "\n")

    # ── Step B: Run phase_find_common ────────────────────────────────────────
    cmd = [
        "phase_find_common",
        inp_file,
        "-WAIT",
        "-NOJOBID",
    ]
    _run_schr(cmd, desc="Phase CPH generation")

    # ── Step C: Locate best hypothesis ───────────────────────────────────────
    phypo_files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".phypo")],
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
        reverse=True,
    )
    if not phypo_files:
        raise FileNotFoundError(
            f"No .phypo files found in {output_dir}. Phase CPH may have failed."
        )

    # ── Step D: Parse survival scores from Phase log ─────────────────────────
    log_file = os.path.join(output_dir, f"{jobname}.log")
    best_hypo, best_score, hypo_table = _parse_phase_hypotheses(log_file, output_dir)

    log.info("Pharmacophore hypotheses generated: %d", len(phypo_files))
    log.info("Best hypothesis: %s  (survival score = %.4f)", best_hypo, best_score)

    if hypo_table is not None:
        hypo_csv = os.path.join(output_dir, "hypothesis_ranking.csv")
        hypo_table.to_csv(hypo_csv, index=False)
        log.info("Hypothesis ranking saved: %s", hypo_csv)

    best_phypo = os.path.join(output_dir, best_hypo + ".phypo") if best_hypo else \
                 os.path.join(output_dir, phypo_files[0])
    return best_phypo


def _parse_phase_hypotheses(log_file: str, output_dir: str):
    """
    Parse the Phase log to extract hypotheses and their survival scores.
    Returns (best_hypo_name, best_score, DataFrame_of_all_hypotheses).
    """
    hypo_records = []
    best_hypo  = None
    best_score = -1.0

    if not os.path.isfile(log_file):
        log.warning("Phase log not found: %s", log_file)
        # Fall back: return first .phypo found
        phypos = [f.replace(".phypo", "")
                  for f in os.listdir(output_dir) if f.endswith(".phypo")]
        return (phypos[0] if phypos else None, best_score, None)

    with open(log_file) as fh:
        for line in fh:
            # Phase log lines typically look like:
            # HYPO_NAME   SURVIVAL_SCORE   FEATURE_STRING
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].endswith(".phypo"):
                name  = parts[0].replace(".phypo", "")
                try:
                    score = float(parts[1])
                except ValueError:
                    continue
                features = parts[2] if len(parts) > 2 else ""
                hypo_records.append({"hypothesis": name, "survival_score": score,
                                      "features": features})
                if score > best_score:
                    best_score = score
                    best_hypo  = name

    hypo_df = pd.DataFrame(hypo_records) if hypo_records else None
    if hypo_df is not None:
        hypo_df = hypo_df.sort_values("survival_score", ascending=False)

    # If log parsing found nothing, use the most recent .phypo
    if best_hypo is None:
        phypos = sorted(
            [f for f in os.listdir(output_dir) if f.endswith(".phypo")],
            key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
            reverse=True,
        )
        if phypos:
            best_hypo = phypos[0].replace(".phypo", "")

    return best_hypo, best_score, hypo_df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Pharmacophore validation (Phase screen + metrics)
# ─────────────────────────────────────────────────────────────────────────────
def validate_pharmacophore(best_phypo: str,
                           screen_mae: str,
                           output_dir: str,
                           jobname: str = "pharm_screen") -> dict:
    """
    Screen the balanced dataset (actives + inactives) against the best
    pharmacophore hypothesis, then compute:
      - ROC-AUC
      - Enrichment Factor (EF) at 1%, 5%, 10%
      - Confusion matrix at default match threshold

    Parameters
    ----------
    best_phypo : Path to the best .phypo file.
    screen_mae : Conformer .mae for the balanced dataset.
    output_dir : Directory to save screen results.
    jobname    : Base name for screen output files.

    Returns
    -------
    Dictionary of performance metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    screen_csv = os.path.join(output_dir, f"{jobname}_hits.csv")

    log.info("Screening dataset against: %s", best_phypo)

    # ── Run phase_screen ─────────────────────────────────────────────────────
    cmd = [
        "phase_screen",
        best_phypo,
        os.path.abspath(screen_mae),
        "-o",      os.path.join(output_dir, jobname),
        "-report", "all",
        "-WAIT",
        "-NOJOBID",
    ]
    _run_schr(cmd, desc="Phase screening")

    # ── Parse screen results ──────────────────────────────────────────────────
    # Phase screen writes a CSV with hit/non-hit info + PhaseScore
    results_file = _find_screen_results(output_dir, jobname)
    if results_file is None:
        raise FileNotFoundError(
            f"No Phase screen results found in {output_dir}."
        )

    results_df = pd.read_csv(results_file)
    log.info("Screen results loaded: %d rows from %s", len(results_df), results_file)

    # Identify activity truth from structure properties in the screen .mae
    truth_df = _load_activities_from_mae(screen_mae)

    # Merge on title / compound ID
    title_col = _detect_title_col(results_df)
    merged = results_df.merge(truth_df, on=title_col, how="left")
    merged["activity"].fillna(0, inplace=True)
    merged["activity"] = merged["activity"].astype(int)

    # Phase screen assigns a score; treat matched compound as predicted positive
    score_col = _detect_score_col(results_df)
    y_true  = merged["activity"].values
    y_score = merged[score_col].fillna(0.0).values

    # Binary prediction: screen match = 1 (Phase marks hits vs non-hits)
    hit_col   = _detect_hit_col(results_df)
    y_pred    = merged[hit_col].fillna(0).astype(int).values if hit_col else (y_score > 0).astype(int)

    metrics = _compute_metrics(y_true, y_score, y_pred)

    # Save merged results
    merged.to_csv(screen_csv, index=False)
    log.info("Screen results (with activity labels) saved: %s", screen_csv)

    _print_metrics(metrics)
    return metrics


def _find_screen_results(output_dir: str, jobname: str) -> str | None:
    """Locate the Phase screen CSV output file."""
    candidates = [
        os.path.join(output_dir, f"{jobname}_hits.csv"),
        os.path.join(output_dir, f"{jobname}.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    csvs = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])
    return os.path.join(output_dir, csvs[0]) if csvs else None


def _load_activities_from_mae(mae_path: str) -> pd.DataFrame:
    """Read activity labels from structure properties in a .mae file."""
    records = []
    for st in StructureReader(mae_path):
        title = st.title
        act   = st.property.get("i_user_activity", None)
        ser   = st.property.get("s_user_series", "")
        records.append({"title": title, "activity": act, "series": ser})
    return pd.DataFrame(records)


def _detect_title_col(df: pd.DataFrame) -> str:
    for c in ["title", "Title", "Name", "name", "compound_id"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _detect_score_col(df: pd.DataFrame) -> str:
    for c in ["PhaseScore", "Score", "phase_score", "score", "Fitness"]:
        if c in df.columns:
            return c
    # Fall back to first numeric column after title
    for c in df.columns[1:]:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[1]


def _detect_hit_col(df: pd.DataFrame):
    for c in ["Hit", "hit", "matched", "Matched", "is_hit"]:
        if c in df.columns:
            return c
    return None


def _compute_metrics(y_true, y_score, y_pred) -> dict:
    """Compute ROC-AUC, enrichment factors, and confusion matrix."""
    n_total   = len(y_true)
    n_actives = int(y_true.sum())

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Enrichment factors at 1%, 5%, 10%
    ef = {}
    sorted_idx = np.argsort(-y_score)
    sorted_true = y_true[sorted_idx]
    for pct in [0.01, 0.05, 0.10]:
        n_top   = max(1, int(np.ceil(n_total * pct)))
        hits    = sorted_true[:n_top].sum()
        random_expected = n_actives * pct
        ef[f"EF_{int(pct*100)}pct"] = hits / random_expected if random_expected > 0 else float("nan")

    return {
        "roc_auc":          roc_auc,
        "confusion_matrix": cm,
        "n_total":          n_total,
        "n_actives":        n_actives,
        "n_inactives":      n_total - n_actives,
        **ef,
    }


def _print_metrics(metrics: dict):
    log.info("─" * 55)
    log.info("VALIDATION METRICS")
    log.info("─" * 55)
    log.info("  ROC-AUC            : %.4f", metrics["roc_auc"])
    log.info("  EF @  1%%           : %.3f", metrics.get("EF_1pct", float("nan")))
    log.info("  EF @  5%%           : %.3f", metrics.get("EF_5pct", float("nan")))
    log.info("  EF @ 10%%           : %.3f", metrics.get("EF_10pct", float("nan")))
    log.info("  Confusion matrix:\n%s", metrics["confusion_matrix"])
    log.info("─" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: filter actives from a .mae and write subset
# ─────────────────────────────────────────────────────────────────────────────
def _write_actives_only(conf_mae: str, actives_out: str):
    """Write a .mae containing only active compounds (i_user_activity==1)."""
    n_written = 0
    with StructureWriter(actives_out) as writer:
        for st in StructureReader(conf_mae):
            if st.property.get("i_user_activity", 0) == 1:
                writer.append(st)
                n_written += 1
    log.info("Filtered actives: %d structures → %s", n_written, actives_out)
    return n_written


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main(df: pd.DataFrame,
         smiles_col: str = "Smiles",
         series_col: str = "Series",
         output_dir: str = OUTPUT_DIR):
    """
    End-to-end pharmacophore pipeline.

    Parameters
    ----------
    df          : Input DataFrame with SMILES and Series columns.
    smiles_col  : Column name for SMILES.
    series_col  : Column name for Series labels.
    output_dir  : Root folder for all outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    log.info("=" * 55)
    log.info("PHARMACOPHORE PIPELINE STARTED")
    log.info("Output directory: %s", os.path.abspath(output_dir))
    log.info("=" * 55)

    # ── STEP 1: Balance dataset ───────────────────────────────────────────────
    log.info("[1/7] Balancing dataset…")
    balanced = balance_dataset(df, smiles_col=smiles_col, series_col=series_col)
    balanced_csv = os.path.join(output_dir, "balanced_dataset.csv")
    balanced.to_csv(balanced_csv, index=False)
    log.info("Balanced dataset saved: %s", balanced_csv)

    # ── STEP 2: SMILES → Schrödinger structures ───────────────────────────────
    log.info("[2/7] Converting SMILES to Schrödinger structures…")
    raw_mae = os.path.join(output_dir, "01_raw_structures.mae")
    n_ok = n_fail = 0
    with StructureWriter(raw_mae) as writer:
        for i, row in balanced.iterrows():
            st = smiles_to_structure(
                smiles   = row[smiles_col],
                title    = f"cpd_{i}",
                activity = int(row["activity"]),
                series   = str(row[series_col]),
            )
            if st is not None:
                writer.append(st)
                n_ok += 1
            else:
                n_fail += 1

    log.info("Structures written: %d  |  Failed: %d  →  %s", n_ok, n_fail, raw_mae)

    # ── STEP 3: LigPrep ───────────────────────────────────────────────────────
    log.info("[3/7] LigPrep: 3D coordinates + OPLS optimization…")
    prepared_mae = os.path.join(output_dir, "02_prepared.mae")
    prepare_ligands(raw_mae, prepared_mae)

    # ── STEP 4: ConfGen ───────────────────────────────────────────────────────
    log.info("[4/7] ConfGen: generating conformers…")
    conformer_mae = os.path.join(output_dir, "03_conformers.mae")
    generate_conformers(prepared_mae, conformer_mae)

    # ── STEP 5: Build pharmacophore (actives only) ────────────────────────────
    log.info("[5/7] Phase CPH: building pharmacophore hypotheses…")
    actives_conf_mae = os.path.join(output_dir, "04_actives_conformers.mae")
    n_actives = _write_actives_only(conformer_mae, actives_conf_mae)
    if n_actives == 0:
        raise ValueError("No active conformers found. Check activity labels in structures.")

    hypo_dir   = os.path.join(output_dir, "hypotheses")
    best_phypo = build_pharmacophore(actives_conf_mae, hypo_dir)
    log.info("Best hypothesis: %s", best_phypo)

    # ── STEP 6: Validate against balanced dataset ─────────────────────────────
    log.info("[6/7] Validating pharmacophore against balanced dataset…")
    screen_dir = os.path.join(output_dir, "screening")
    metrics    = validate_pharmacophore(best_phypo, conformer_mae, screen_dir)

    # ── STEP 7: Summary ───────────────────────────────────────────────────────
    log.info("[7/7] Saving final summary…")
    summary = {
        "best_hypothesis":    best_phypo,
        "n_actives":          int(balanced["activity"].sum()),
        "n_inactives":        int((balanced["activity"] == 0).sum()),
        "roc_auc":            metrics["roc_auc"],
        "EF_1pct":            metrics.get("EF_1pct"),
        "EF_5pct":            metrics.get("EF_5pct"),
        "EF_10pct":           metrics.get("EF_10pct"),
        "raw_structures":     raw_mae,
        "prepared_mae":       prepared_mae,
        "conformers_mae":     conformer_mae,
        "hypothesis_dir":     hypo_dir,
        "screen_dir":         screen_dir,
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(output_dir, "pipeline_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    log.info("=" * 55)
    log.info("PIPELINE COMPLETE")
    log.info("  Best hypothesis : %s", best_phypo)
    log.info("  ROC-AUC         : %.4f", metrics["roc_auc"])
    log.info("  EF @ 1%%         : %.3f", metrics.get("EF_1pct", float("nan")))
    log.info("  EF @ 5%%         : %.3f", metrics.get("EF_5pct", float("nan")))
    log.info("  EF @ 10%%        : %.3f", metrics.get("EF_10pct", float("nan")))
    log.info("  Output dir      : %s", os.path.abspath(output_dir))
    log.info("=" * 55)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point – replace the demo df below with your actual dataframe
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── If running standalone, load the QSAR Excel file directly ─────────────
    # Replace this block with `df = your_dataframe` if calling from a notebook.
    try:
        import pandas as pd
        EXCEL = "TB Project QSAR.xlsx"
        df_dict = pd.read_excel(EXCEL, sheet_name=None)
        sheets  = list(df_dict.values())

        Series_A = sheets[2].copy(); Series_A["Series"] = "Series A"
        Series_B = sheets[3].copy(); Series_B["Series"] = "Series B"
        Series_C = sheets[4].copy(); Series_C["Series"] = "Series C"
        Series_D = sheets[5].copy(); Series_D["Series"] = "Series D"
        Series_E = sheets[6].copy(); Series_E["Series"] = "Series E"

        df_all = pd.concat([Series_A, Series_B, Series_C, Series_D, Series_E],
                            ignore_index=True)

        # non_numeric inactives = rows where IC50 is not a number
        df_all["_ic50_num"] = pd.to_numeric(df_all["IC50 uM"], errors="coerce")
        df_all.loc[df_all["_ic50_num"].isna(), "Series"] = "non_numeric"
        df_all.drop(columns=["_ic50_num"], inplace=True)

        # Use the canonical SMILES column; rename for clarity
        df_all.rename(columns={"Canonical SMILES.1": "Smiles"}, inplace=True)
        df_all = df_all.dropna(subset=["Smiles"])

        log.info("Loaded %d compounds from %s", len(df_all), EXCEL)

    except Exception as e:
        log.error("Could not load data: %s", e)
        sys.exit(1)

    main(df_all, smiles_col="Smiles", series_col="Series")
