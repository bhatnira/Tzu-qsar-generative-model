ARCHIVE FOLDER GUIDE
====================

This folder stores repository items that were moved out of the project root to
keep the active workspace cleaner.

CURRENT SUBFOLDERS
------------------
root_level_legacy/
  Items previously stored directly at the repository root.

  csv_generation_exports/
    Legacy generation/scoring CSV exports that were cluttering the root.

  notebook_duplicates/
    Archived root-level notebook duplicates. These point to the canonical
    notebooks stored in notebooks/.

  raw_downloads/
    Large downloaded files moved out of the root.

WHAT IS STILL ACTIVE?
---------------------
Use these active locations instead of the archive:
- notebooks/ for notebooks
- outputs/ for current generated results
- docs/ for documentation
- reinvent_integration/ for workflow code

WHY THIS EXISTS
---------------
The archive keeps older or duplicate root-level files available without mixing
them into the active working area of the repository.
