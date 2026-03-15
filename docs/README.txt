QSAR PROJECT TZU — REPOSITORY GUIDE
===================================

This repository is now organized around a target-free QSAR + REINVENT workflow.
All files are inside folders — there are no loose files at the repository root.

TOP-LEVEL FOLDERS
-----------------
qsar_core/
  Core Python modules: data_loader, descriptors, model, clustering,
  visualization, and shared path constants.

scripts/
  All runnable scripts (pipeline runner, prediction generators, animation
  generators, test/debug scripts). Run from the repository root.

config/
  Environment configuration files: config/environment.yml, config/requirements.txt,
  config/setup_env.sh.

data/raw/
  Primary dataset: TB Project QSAR.xlsx.

reinvent_integration/
  Main generation, scoring, and target-free ranking workflow.

outputs/
  Generated results, ranked candidate tables, Excel workbooks, and summaries.

docs/
  Plain-text and markdown documentation.

archive/
  Root-level legacy files and duplicate notebooks moved out of the active root.

notebooks/
  Jupyter notebooks for debugging and exploratory analysis.

presentation/
  LaTeX slide deck assets.

external_tools/
  External or vendor-specific integrations.

HOW TO RUN (from repo root)
----------------------------
# Export/rebuild the QSAR artifact:
  python reinvent_integration/export_qsar_model.py

# Run the training pipeline:
  python scripts/run_pipeline.py

# Run generation with a preset:
  python scripts/run_pipeline.py --generation-preset balanced

# Rebuild Gate 1/2/3 ranked outputs:
  python reinvent_integration/run_hybrid_stage_gates.py

# Quick smoke-test:
  python scripts/test_import.py

# Dataset check:
  python scripts/test_data.py

MOST IMPORTANT OUTPUT FILES
----------------------------
outputs/hybrid_stage/gate1/gate1_shortlist.xlsx
  Original top-200 scaffold-diverse shortlist.

outputs/hybrid_stage/gate3_target_free_translational/translational_candidates.xlsx
  Final target-free reranked 200-candidate workbook.

CURRENT TARGET-FREE PIPELINE
----------------------------
Gate 1:
  QSAR + ADME multi-parameter triage + scaffold diversity.

Gate 2:
  QSAR uncertainty + synthetic accessibility + liability filters + ligand-based similarity.

Gate 3:
  Translational prioritization with toxicity, DDI, absorption, and experimental priority score.

LEGACY OUTPUTS
--------------
Older target-based Schrödinger and mechanistic handoff folders were moved to:
  outputs/hybrid_stage/legacy/

ROOT CLEANUP
------------
Loose root-level generation CSVs, duplicate notebooks, and large downloaded
files were moved to:
  archive/root_level_legacy/

DOCUMENTATION TO READ
---------------------
docs/TERMS_AND_WORKFLOW_EXPLAINED.txt
  Comprehensive report (.txt): full glossary, terms, pipeline, scoring,
  and rationale.

docs/REPO_STRUCTURE_AND_OUTPUTS.txt
  Practical map of all folders and where the current outputs live.

outputs/hybrid_stage/README.txt
  Output-specific guide for Gate 1 / Gate 2 / Gate 3 artifacts.

archive/README.txt
  Explains what was moved out of the root and where to find it.
