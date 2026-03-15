"""
Train and export the best QSAR model from the current project pipeline,
so REINVENT4 can call it during generation.

Usage:
    python reinvent_integration/export_qsar_model.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_core.data_loader import (
    load_excel_sheets,
    apply_smiles_cleaning,
    combine_and_deduplicate,
    filter_numeric_ic50,
)
from qsar_core.descriptors import compute_descriptors
from qsar_core.model import train_and_select
from qsar_core.paths import DATASET_XLSX


def main() -> None:
    project_root = PROJECT_ROOT
    artifacts_dir = project_root / "reinvent_integration" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    excel_path = DATASET_XLSX
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing dataset: {excel_path}")

    series_dfs = load_excel_sheets(str(excel_path))
    series_dfs = apply_smiles_cleaning(series_dfs)
    full_df = combine_and_deduplicate(series_dfs)
    numeric_df = filter_numeric_ic50(full_df)
    numeric_df["transformed_IC50"] = np.log10(numeric_df["IC50 uM"] + 1e-8)

    train_df = numeric_df[numeric_df["Series_Code"].isin(["A", "B", "D"])].reset_index(drop=True)
    test_df = numeric_df[numeric_df["Series_Code"] == "C"].reset_index(drop=True)

    smiles_train = train_df["Canonical_SMILES"].values
    smiles_test = test_df["Canonical_SMILES"].values
    y_train = train_df["transformed_IC50"].values
    y_test = test_df["transformed_IC50"].values

    X_train_raw = compute_descriptors(smiles_train)
    X_test_raw = compute_descriptors(smiles_test)

    trained_models, imputers, scalers, best_desc, best_model_name, results_df = train_and_select(
        X_train_raw, y_train, X_test_raw, y_test
    )

    model_obj = trained_models[(best_desc, best_model_name)]
    imputer_obj = imputers[best_desc]
    scaler_obj = scalers.get((best_desc, best_model_name), None)

    ranked_results = results_df.sort_values("R2", ascending=False).reset_index(drop=True)
    positive_top = ranked_results[ranked_results["R2"] > 0].head(5)
    ensemble_models = []
    for _, row in positive_top.iterrows():
        desc_name = str(row["Descriptor"])
        model_name = str(row["Model"])
        if desc_name == best_desc and model_name == best_model_name:
            continue
        ensemble_models.append(
            {
                "descriptor_name": desc_name,
                "model_name": model_name,
                "r2": float(row["R2"]),
                "imputer": imputers[desc_name],
                "scaler": scalers.get((desc_name, model_name), None),
                "model": trained_models[(desc_name, model_name)],
            }
        )

    model_bundle = {
        "model": model_obj,
        "imputer": imputer_obj,
        "scaler": scaler_obj,
        "descriptor_name": best_desc,
        "model_name": best_model_name,
        "best_r2": float(ranked_results["R2"].max()),
        "ensemble_models": ensemble_models,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "notes": "Predicts log10(IC50_uM)",
    }

    bundle_path = artifacts_dir / "qsar_best_model.joblib"
    metrics_path = artifacts_dir / "qsar_model_benchmark.csv"
    meta_path = artifacts_dir / "qsar_model_meta.json"

    joblib.dump(model_bundle, bundle_path)
    results_df.sort_values("R2", ascending=False).to_csv(metrics_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bundle": str(bundle_path),
                "descriptor_name": best_desc,
                "model_name": best_model_name,
                "best_r2": float(results_df["R2"].max()),
                "ensemble_models": [
                    {
                        "descriptor_name": str(x["Descriptor"]),
                        "model_name": str(x["Model"]),
                        "r2": float(x["R2"]),
                    }
                    for _, x in positive_top.iterrows()
                ],
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
            },
            handle,
            indent=2,
        )

    print(f"Saved QSAR bundle: {bundle_path}")
    print(f"Saved benchmark CSV: {metrics_path}")
    print(f"Saved metadata JSON: {meta_path}")


if __name__ == "__main__":
    main()
