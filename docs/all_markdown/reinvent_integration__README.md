# REINVENT4 Integration (QSAR + Pharmacophore Guided Generation)

This folder adds REINVENT4-compatible guidance using:
- QSAR prediction from this project
- Optional Schrödinger Phase pharmacophore reward computed as:
  1) build 3D structures with LigPrep
  2) build multiple conformers with ConfGen
  3) align conformers to ligand-based pharmacophore with Phase
  4) select best conformer fit score per molecule

It supports staged learning templates for:
- REINVENT
- LibInvent
- LinkInvent
- Mol2Mol

## Files

- `export_qsar_model.py`:
  Trains and exports the best local QSAR model bundle.

- `prepare_series_ce_data.py`:
  Extracts Series C/E compounds to SMILES files for transfer learning and similarity guidance.

- `reinvent_qsar_pharm_score.py`:
  External scorer script for REINVENT `ExternalProcess` component.
  Reads SMILES from stdin and emits JSON payload to stdout.

- `configs/*.toml`:
  Starter staged-learning TOML templates for all 4 model families.

- `configs/reinvent_transfer_series_ce.toml`:
  Transfer-learning template to fine-tune REINVENT on Series C/E compounds.

- `configs/reinvent_staged_series_ce_guided.toml`:
  Guided staged-learning template that combines QSAR + pharmacophore + C/E similarity.

## 0) Prepare Series C/E data

Run from project root:

```bash
python reinvent_integration/prepare_series_ce_data.py
```

Outputs:
- `reinvent_integration/data/series_ce_smiles.smi`
- `reinvent_integration/data/series_ce_unique.smi`
- `reinvent_integration/data/series_ce_labeled.csv`

## 1) Export the QSAR model bundle

Run from project root:

```bash
python reinvent_integration/export_qsar_model.py
```

Artifacts are written to:
- `reinvent_integration/artifacts/qsar_best_model.joblib`
- `reinvent_integration/artifacts/qsar_model_benchmark.csv`
- `reinvent_integration/artifacts/qsar_model_meta.json`

## 2) Install / prepare REINVENT4

Clone and install REINVENT4 in your chosen environment (CPU/GPU) following upstream instructions:
- https://github.com/MolecularAI/REINVENT4

Expected command shape:

```bash
reinvent -l run.log path/to/config.toml
```

## 3) Update a TOML template

Pick one config from `reinvent_integration/configs/` and update:

- `prior_file` (family-specific prior/agent model)
- `smiles_file` (required for LibInvent/LinkInvent/Mol2Mol)
- `params.executable` and `params.args`
  - keep them pointing to `reinvent_qsar_pharm_score.py`
  - ensure the Python interpreter can import this project modules
- `--pharm-hypo` path if you want Phase contribution
- `--schrodinger` installation path if using Phase scoring

## 4) Run staged learning

```bash
reinvent -l reinvent_run.log reinvent_integration/configs/reinvent_staged_qsar_pharm.toml
```

Use the corresponding config for LibInvent/LinkInvent/Mol2Mol.

## 5) Fine-tune on Series C/E and generate alike molecules

1. Fine-tune the prior on Series C/E compounds:

```bash
reinvent -l tl_series_ce.log reinvent_integration/configs/reinvent_transfer_series_ce.toml
```

2. Run guided staged learning from the fine-tuned model:

```bash
reinvent -l rl_series_ce.log reinvent_integration/configs/reinvent_staged_series_ce_guided.toml
```

This setup reduces randomness by combining:
- transfer learning on C/E compounds
- inception memory seeded with C/E compounds
- explicit similarity reward (`sim_score`) to C/E references

## Scoring details

The scorer returns these payload keys:
- `qsar_pharm_score` (main optimization objective)
- `qsar_score`
- `pharm_score`
- `pharm_fit_score`
- `pharm_rmsd`
- `pharm_matched_features`
- `sim_score`
- `required_match`
- `pred_pic50`

Combined score is a weighted mean:

\[
\text{combined} = \frac{w_q \cdot s_q + w_p \cdot s_p}{w_q + w_p}
\]

Where defaults are:
- `w_q = 0.7`
- `w_p = 0.3`

## Notes

- If Phase execution is unavailable or fails, pharmacophore score falls back to 0.0.
- QSAR prediction still runs, so generation remains guided.
- Tune `--target-pic50`, `--qsar-weight`, `--pharm-weight`, `--pharm-center`, and `--pharm-scale` in `params.args` for your campaign.
- Tune conformer generation with `--confgen-max` and `--confgen-energy-window`.
- Tune C/E similarity guidance with `--ref-smiles-file`, `--sim-weight`, `--sim-radius`, and `--sim-nbits`.
- Preserve a mandatory motif with `--required-smarts` and `--required-weight`.
