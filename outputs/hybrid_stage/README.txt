HYBRID STAGE OUTPUTS
====================

This folder contains the current best candidate outputs for the project.

CURRENT ACTIVE FOLDERS
----------------------
gate1/
  Original top-200 scaffold-diverse shortlist from QSAR + ADME triage.

gate2_target_free_refinement/
  Same 200 compounds reranked with:
  - QSAR uncertainty from up to five positive-R² models
  - synthetic accessibility
  - PAINS/BRENK/NIH alerts
  - ligand-based 2D similarity
  - ligand-based 3D shape similarity

gate3_target_free_translational/
  Final target-free reranking with:
  - tox risk
  - DDI / CYP inhibition risk
  - absorption support
  - experimental priority score

LEGACY FOLDERS
--------------
legacy/
  Older target-based handoff outputs preserved for reference only.

WHICH FILE TO OPEN
------------------
Original top 200:
  gate1/gate1_shortlist.xlsx

Final best 200 under current target-free workflow:
  gate3_target_free_translational/translational_candidates.xlsx

CURRENT COUNTS
--------------
Gate 1 shortlist: 200
Gate 2 passes:    46
Gate 3 passes:    34

IMPORTANT NOTE
--------------
Use the target-free folders as the current workflow of record.
The legacy folders are historical and not the preferred path anymore.
