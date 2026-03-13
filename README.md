# QSAR Project — Setup & Run

Quick steps to create a reproducible conda environment and run the pipeline.

- Edit `main.py` and set the Excel filename variable `file_name` to your data file (e.g. "TB Project QSAR.xlsx").
- Create the conda environment (uses `environment.yml`):

```bash
./setup_env.sh
# or
conda env create -f environment.yml -n qsar-env
```

- Activate the environment and run the pipeline:

```bash
conda activate qsar-env
python main.py
```

Notes:
- Use the conda-forge channel to avoid RDKit / NumPy ABI mismatches.
- UMAP / HDBSCAN steps may trigger long JIT compilation (numba/pynndescent); do not interrupt the run until it finishes.
- Outputs (CSVs and PNGs) are written to the `outputs/` folder when the pipeline completes.

If you'd like, I can also create an `environment.yml` variant pinned to exact versions or add a small wrapper to run `main.py` with timing/logging.

## REINVENT4 Guided Generation (QSAR + Pharmacophore)

A REINVENT4 integration is available in `reinvent_integration/` for guided generation using:
- exported QSAR model predictions
- optional Schrödinger Phase pharmacophore scoring

See:
- `reinvent_integration/README.md` for setup and run instructions
- `reinvent_integration/configs/` for staged-learning templates for REINVENT, LibInvent, LinkInvent, and Mol2Mol
