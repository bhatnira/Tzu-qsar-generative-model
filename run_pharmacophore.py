"""
run_pharmacophore.py
Wrapper to run pharmacophore_pipeline.py with explicit file-based logging.
Run with: $SCHRODINGER/run python run_pharmacophore.py
"""
import sys
import traceback
import logging

# Set up file + console logging BEFORE importing anything else
LOG_FILE = "pharm_pipeline.log"

fh = logging.FileHandler(LOG_FILE, mode="w")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
fh.setFormatter(fmt)
ch.setFormatter(fmt)

root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(fh)
root.addHandler(ch)

log = logging.getLogger("runner")
log.info("Starting pharmacophore pipeline runner…")

try:
    # Import pipeline
    log.info("Importing pharmacophore_pipeline…")
    import pharmacophore_pipeline as pp
    log.info("Import OK, loading data…")

    import pandas as pd
    # Use pre-converted CSV (Schrödinger's openpyxl is too old for .xlsx)
    CSV_FILE = "pharm_input.csv"
    df_all = pd.read_csv(CSV_FILE)
    df_all = df_all.dropna(subset=["Smiles"])

    log.info("Loaded %d compounds from %s", len(df_all), CSV_FILE)
    log.info("Series value_counts:\n%s", df_all["Series"].value_counts().to_string())

    # Run pipeline
    metrics = pp.main(df_all, smiles_col="Smiles", series_col="Series")
    log.info("Pipeline returned metrics: %s", metrics)

except Exception:
    log.error("Pipeline FAILED:\n%s", traceback.format_exc())
    sys.exit(1)

log.info("Done.")
