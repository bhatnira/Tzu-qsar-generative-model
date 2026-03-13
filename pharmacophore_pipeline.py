"""
pharmacophore_pipeline.py
=========================
Ligand-based pharmacophore modeling pipeline using the Schrödinger Python API.
Uses LOCAL compute only — all 24 cores, no license-token checkout from job server.

Run with:
    $SCHRODINGER/run python pharmacophore_pipeline.py

Parallelism (all local, no remote job server / token consumption):
    - LigPrep:          -NJOBS 24 -HOST localhost -NOJOBID
    - ConfGen:          -NJOBS 24 -HOST localhost -NOJOBID
    - Phase find_common: -HOST localhost:24 -NOJOBID
    - Phase screen:     -HOST localhost:24 -NJOBS 24 -NOJOBID

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
NCPUS        = os.cpu_count() or 24   # use all available cores (24 on this workstation)
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
        "-NJOBS", str(NCPUS),    # split across all cores
        "-HOST",  "localhost",   # run locally (no remote job server)
        "-WAIT",                 # block until complete
        "-NOJOBID",              # no job ID registration = no token checkout
    ]
    log.info("LigPrep using %d local CPU cores", NCPUS)
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

    # ConfGen takes a structure file as a positional argument with CLI flags
    cmd = [
        "confgenx",
        os.path.abspath(input_mae),
        "-m",         str(max_confs),
        "-optimize",
        "-force_field", "OPLS_2005",
        "-NJOBS", str(NCPUS),    # split across all cores
        "-HOST",  "localhost",   # run locally (no remote job server)
        "-WAIT",
        "-NOJOBID",              # no job ID = no token checkout
    ]
    log.info("ConfGen using %d local CPU cores", NCPUS)
    _run_schr(cmd, desc="ConfGen")

    # ConfGen writes output with -out.maegz suffix, either next to input or in CWD
    input_basename = os.path.splitext(os.path.basename(input_mae))[0]
    input_base_abs = os.path.splitext(os.path.abspath(input_mae))[0]
    candidates = [
        input_base_abs + "-out.maegz",
        input_base_abs + "-out.mae",
        # ConfGen may also write to CWD based on filename only
        os.path.join(os.getcwd(), input_basename + "-out.maegz"),
        os.path.join(os.getcwd(), input_basename + "-out.mae"),
    ]
    confgen_output = None
    for c in candidates:
        if os.path.isfile(c):
            confgen_output = c
            break

    if confgen_output is None:
        # Search CWD and input dir for any *-out* files
        search_dirs = set([os.getcwd(), os.path.dirname(os.path.abspath(input_mae))])
        for d in search_dirs:
            all_files = [os.path.join(d, f) for f in os.listdir(d)
                         if f.endswith((".mae", ".maegz")) and "out" in f.lower()]
            if all_files:
                confgen_output = sorted(all_files, key=os.path.getmtime)[-1]
                break

    if confgen_output is None:
        raise FileNotFoundError(
            f"ConfGen produced no output. Expected {candidates[0]} or similar."
        )

    # Move/rename to our expected output path
    if confgen_output != os.path.abspath(output_mae):
        shutil.copy2(confgen_output, output_mae)
        log.info("Copied ConfGen output: %s → %s", confgen_output, output_mae)

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
      a) Create a Phase project from active conformers.
      b) Archive the project to .phzip.
      c) Run phase_find_common to identify common pharmacophore hypotheses.
      d) Parse results and return the best hypothesis.

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

    # ── Step A: Create Phase project ─────────────────────────────────────────
    project_dir = os.path.join(output_dir, f"{jobname}.phprj")
    if os.path.isdir(project_dir):
        shutil.rmtree(project_dir)

    cmd_import = [
        os.path.join(SCHRODINGER, "utilities", "phase_project"),
        project_dir,
        "import",
        "-i", os.path.abspath(actives_mae),
        "-new",
        "-act", "i_user_activity",
        "-fmt", "single",
    ]
    log.info("Creating Phase project: %s", " ".join(cmd_import))
    result = subprocess.run(cmd_import, capture_output=True, text=True,
                            env={**os.environ, "SCHRODINGER": SCHRODINGER})
    if result.returncode != 0:
        log.error("phase_project import STDOUT:\n%s", result.stdout[-2000:])
        log.error("phase_project import STDERR:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"Phase project import failed (exit {result.returncode})")
    log.info("Phase project created: %s", project_dir)

    # ── Step B: Create pharmacophore sites ───────────────────────────────────
    cmd_sites = [
        os.path.join(SCHRODINGER, "utilities", "phase_project"),
        project_dir,
        "revise",
        "-sites",
    ]
    log.info("Creating pharmacophore sites: %s", " ".join(cmd_sites))
    result = subprocess.run(cmd_sites, capture_output=True, text=True,
                            env={**os.environ, "SCHRODINGER": SCHRODINGER})
    if result.returncode != 0:
        log.error("phase_project revise -sites STDOUT:\n%s", result.stdout[-2000:])
        log.error("phase_project revise -sites STDERR:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"Phase site creation failed (exit {result.returncode})")
    log.info("Pharmacophore sites created for project: %s", project_dir)

    # ── Step C: Archive to .phzip ────────────────────────────────────────────
    cmd_archive = [
        os.path.join(SCHRODINGER, "utilities", "phase_project"),
        project_dir,
        "archive",
    ]
    log.info("Archiving Phase project: %s", " ".join(cmd_archive))
    result = subprocess.run(cmd_archive, capture_output=True, text=True,
                            env={**os.environ, "SCHRODINGER": SCHRODINGER})
    if result.returncode != 0:
        log.error("phase_project archive STDOUT:\n%s", result.stdout[-2000:])
        log.error("phase_project archive STDERR:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"Phase project archive failed (exit {result.returncode})")

    phzip = project_dir.replace(".phprj", ".phzip")
    if not os.path.isfile(phzip):
        # phase_project archive often writes .phzip to CWD, not next to .phprj
        candidates = [
            os.path.join(output_dir, f"{jobname}.phzip"),
            os.path.join(os.getcwd(), f"{jobname}.phzip"),
            f"{jobname}.phzip",
        ]
        found = False
        for cand in candidates:
            if os.path.isfile(cand):
                shutil.move(cand, phzip)
                log.info("Moved .phzip from %s → %s", cand, phzip)
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Expected .phzip not found at {phzip} or CWD. "
                f"Checked: {candidates}"
            )
    log.info("Phase archive: %s", phzip)

    # ── Step C: Run phase_find_common ────────────────────────────────────────
    cmd_find = [
        os.path.join(SCHRODINGER, "phase_find_common"),
        phzip,
        "-sites", f"{min_sites}:{max_sites}",
        "-HOST",  f"localhost:{NCPUS}",  # use all cores locally
        "-LOCAL",                # run locally, no remote job server
        "-WAIT",
    ]
    log.info("Phase find_common using %d local CPU cores", NCPUS)
    log.info("Running phase_find_common: %s", " ".join(cmd_find))
    result = subprocess.run(cmd_find, capture_output=True, text=True,
                            env={**os.environ, "SCHRODINGER": SCHRODINGER})
    if result.returncode != 0:
        log.error("phase_find_common STDOUT:\n%s", result.stdout[-2000:])
        log.error("phase_find_common STDERR:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"Phase find_common failed (exit {result.returncode})")

    # ── Step D: Locate results ───────────────────────────────────────────────
    # phase_find_common outputs: <jobName>_find_common.zip
    fc_zip = os.path.join(output_dir, f"{jobname}_find_common.zip")
    if not os.path.isfile(fc_zip):
        # Try current directory
        fc_zip_alt = f"{jobname}_find_common.zip"
        if os.path.isfile(fc_zip_alt):
            shutil.move(fc_zip_alt, fc_zip)
        else:
            # Search for any _find_common.zip
            import glob
            candidates = glob.glob(os.path.join(output_dir, "*_find_common.zip")) + \
                         glob.glob("*_find_common.zip")
            if candidates:
                fc_zip = candidates[0]
            else:
                raise FileNotFoundError(f"No _find_common.zip found after phase_find_common.")

    # Unzip results
    fc_dir = os.path.join(output_dir, f"{jobname}_find_common")
    import zipfile
    with zipfile.ZipFile(fc_zip, "r") as zf:
        zf.extractall(fc_dir)
    log.info("Extracted find_common results to: %s", fc_dir)

    # Some Phase archives contain a single top-level directory with the same
    # name as the archive stem. Descend into it if present.
    fc_root = fc_dir
    nested_dir = os.path.join(fc_dir, os.path.basename(fc_dir))
    if os.path.isdir(nested_dir):
        fc_root = nested_dir
        log.info("Using nested find_common results directory: %s", fc_root)

    # Find .phypo files in hypotheses subdirectory
    hypo_subdir = os.path.join(fc_root, "hypotheses")
    if not os.path.isdir(hypo_subdir):
        hypo_subdir = fc_root  # fallback: hypotheses at root level

    phypo_files = sorted(
        [f for f in os.listdir(hypo_subdir) if f.endswith(".phypo")]
    )

    if not phypo_files:
        raise FileNotFoundError(
            f"No .phypo files found in {hypo_subdir}. Phase CPH may have failed."
        )

    # Parse scores.csv if it exists
    scores_csv = os.path.join(fc_root, "scores.csv")
    best_hypo = None
    best_score = -1.0

    if os.path.isfile(scores_csv):
        scores_df = pd.read_csv(scores_csv)
        log.info("Phase scores:\n%s", scores_df.to_string())
        scores_df.to_csv(os.path.join(output_dir, "hypothesis_ranking.csv"), index=False)
        # Pick hypothesis with highest Survival score
        if "Survival" in scores_df.columns:
            idx = scores_df["Survival"].idxmax()
            best_hypo = scores_df.loc[idx, "ID"] if "ID" in scores_df.columns else phypo_files[0].replace(".phypo", "")
            best_score = scores_df.loc[idx, "Survival"]
    
    if best_hypo is None:
        # Use first hypothesis file
        best_hypo = phypo_files[0].replace(".phypo", "")

    best_phypo = os.path.join(hypo_subdir, best_hypo + ".phypo")
    if not os.path.isfile(best_phypo):
        best_phypo = os.path.join(hypo_subdir, phypo_files[0])

    log.info("Pharmacophore hypotheses generated: %d", len(phypo_files))
    log.info("Best hypothesis: %s  (survival score = %.4f)", best_hypo, best_score)

    return best_phypo


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
    # Syntax: phase_screen <source> <hypo> <jobname> [options]
    screen_job = os.path.join(output_dir, jobname)
    cmd = [
        os.path.join(SCHRODINGER, "phase_screen"),
        os.path.abspath(screen_mae),
        os.path.abspath(best_phypo),
        screen_job,
        "-distinct",
        "-match", "3",                    # allow partial 3-site matches (less strict than default all-sites)
        "-HOST",  f"localhost:{NCPUS}",  # use all cores locally
        "-NJOBS", str(NCPUS),              # split screening across cores
    ]
    log.info("Phase screen using %d local CPU cores", NCPUS)
    log.info("Running phase_screen: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env={**os.environ, "SCHRODINGER": SCHRODINGER})
    if result.returncode != 0:
        log.error("phase_screen STDOUT:\n%s", result.stdout[-2000:])
        log.error("phase_screen STDERR:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"Phase screening failed (exit {result.returncode})")

    # ── Parse screen results ──────────────────────────────────────────────────
    # phase_screen outputs <jobname>-hits.maegz with matched structures + PhaseScreenScore
    hits_file = screen_job + "-hits.maegz"
    if not os.path.isfile(hits_file):
        hits_file = screen_job + "-hits.mae"
    if not os.path.isfile(hits_file):
        # Search for any hits file
        import glob
        hits_candidates = glob.glob(os.path.join(output_dir, f"{jobname}*hits*"))
        if hits_candidates:
            hits_file = hits_candidates[0]
        else:
            log.warning("No phase_screen hits file found. Assuming no matches.")
            hits_file = None

    # Load all screened compounds with their true activities
    truth_df = _load_activities_from_mae(screen_mae)  # title, activity, series
    log.info("Loaded truth labels for %d structures", len(truth_df))

    # Parse hits to get matched titles + scores
    hit_titles = set()
    hit_scores = {}
    if hits_file and os.path.isfile(hits_file):
        for st in StructureReader(hits_file):
            hit_titles.add(st.title)
            score = st.property.get("r_phase_PhaseScreenScore",
                    st.property.get("r_phase_Fitness", 0.0))
            hit_scores[st.title] = score
        log.info("phase_screen matched %d structures", len(hit_titles))
    else:
        log.warning("No hits found from phase_screen.")

    # Build merged dataframe: all compounds with hit/no-hit prediction
    truth_df["y_pred"] = truth_df["title"].apply(lambda t: 1 if t in hit_titles else 0)
    truth_df["y_score"] = truth_df["title"].apply(lambda t: hit_scores.get(t, 0.0))
    truth_df["activity"] = truth_df["activity"].fillna(0).astype(int)

    y_true  = truth_df["activity"].values
    y_score = truth_df["y_score"].values
    y_pred  = truth_df["y_pred"].values

    metrics = _compute_metrics(y_true, y_score, y_pred)

    # Save merged results
    screen_csv = os.path.join(output_dir, f"{jobname}_results.csv")
    truth_df.to_csv(screen_csv, index=False)
    log.info("Screen results saved: %s", screen_csv)

    _print_metrics(metrics)
    return metrics


def _load_activities_from_mae(mae_path: str) -> pd.DataFrame:
    """Read activity labels from structure properties in a .mae file."""
    records = []
    for st in StructureReader(mae_path):
        title = st.title
        act   = st.property.get("i_user_activity", None)
        ser   = st.property.get("s_user_series", "")
        records.append({"title": title, "activity": act, "series": ser})
    return pd.DataFrame(records)


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
