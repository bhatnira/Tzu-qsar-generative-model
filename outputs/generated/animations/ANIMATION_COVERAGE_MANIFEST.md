# Animation Coverage Manifest

This manifest explains what each generated image/GIF contains.

## Global overview

- `all_generation_folders_big_timeline_first_frame.png`
  - Type: static preview frame
  - Coverage: overview of 7 generation folders
  - Purpose: first frame only, not exhaustive by category

- `all_generation_folders_big_timeline.gif`
  - Type: animated overview
  - Coverage: overview across 7 generation folders and 3 modes (`Plain`, `RL`, `CL`)
  - Purpose: synchronized folder-level wallboard, not a category-exhaustive scaffold/bioisostere animation

## Representative single-example files

These are curated examples used to illustrate the concept with one selected compound path.

- `final_structures/minimal_diffs/spiral_preservation_big.gif`
  - Type: representative example
  - Coverage: 1 preserved-scaffold example

- `final_structures/minimal_diffs/scaffold_preservation_minimal.gif`
  - Type: representative example
  - Coverage: 1 preserved-scaffold example

- `final_structures/minimal_diffs/scaffold_non_preservation_minimal.gif`
  - Type: representative example
  - Coverage: 1 non-preserved-scaffold example

- `final_structures/minimal_diffs/non_spiral_preservation_big.gif`
  - Type: representative example
  - Coverage: 1 non-preserved-scaffold example

- `final_structures/minimal_diffs/minimal_combined_summary.gif`
  - Type: representative summary
  - Coverage: 3 representative examples total
    - 1 preserved
    - 1 non-preserved
    - 1 bioisostere-like replacement

- `final_structures/minimal_diffs/biostere_replacement_big.gif`
  - Type: representative example
  - Coverage: 1 bioisostere-like replacement example
  - Note: typo-compatible alias of the bioisostere big view

- `final_structures/minimal_diffs/bioisostere_replacement_minimal.gif`
  - Type: representative example
  - Coverage: 1 bioisostere-like replacement example

- `final_structures/minimal_diffs/bioisostere_replacement_big.gif`
  - Type: representative example
  - Coverage: 1 bioisostere-like replacement example

## Exhaustive all-compounds files

These include every valid unique input compound found for the category that had complete `Plain`, `RL`, and `CL` paths.

- `final_structures/minimal_diffs/spiral_preservation_all_compounds_part_01.gif`
  - Type: exhaustive category animation
  - Coverage: all valid preserved-scaffold compounds
  - Count: 5 of 5

- `final_structures/minimal_diffs/non_spiral_preservation_all_compounds_part_01.gif`
  - Type: exhaustive category animation
  - Coverage: all valid non-preserved-scaffold compounds
  - Count: 13 of 13

- `final_structures/minimal_diffs/bioisostere_replacement_all_compounds_part_01.gif`
  - Type: exhaustive category animation
  - Coverage: all valid bioisostere-replacement compounds
  - Count: 4 of 4

## Interpretation

If the goal is **all compounds**, use the `*_all_compounds_part_01.gif` files.

If the goal is **clean presentation examples**, use the `*_minimal.gif`, `*_big.gif`, or `minimal_combined_summary.gif` files.
