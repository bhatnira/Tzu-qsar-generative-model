Gate 3 target-free translational prioritization

Generated files:
- translational_candidates.csv: final reranked candidates for experimental planning
- translational_candidates.xlsx: same data with structures and analysis
- experimental_shortlist_template.csv: wet-lab planning template

Current status:
- 200 total candidates carried into Gate 3
- 34 candidates currently flagged as passes_gate3_translational = 1

Important columns:
- experimental_priority_score: final score for lab prioritization
- gate3_translational_score: combined translational score before final ranking
- tox_safety_score: aggregated toxicity safety score
- ddi_safety_score: CYP/DDI safety score
- absorption_score: target-free exposure support score
- passes_gate3_translational: recommended experimental shortlist flag

Implemented target-free translational signals:
1) Tox liabilities: hERG, DILI, ClinTox, AMES, carcinogenicity
2) DDI liabilities: CYP1A2/2C19/2C9/2D6/3A4 inhibition risk
3) Absorption support: HIA, bioavailability, PAMPA, Caco2
4) Final experimental priority score for lab selection
