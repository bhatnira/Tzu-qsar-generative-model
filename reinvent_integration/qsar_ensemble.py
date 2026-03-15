from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from rdkit import Chem

from qsar_core.descriptors import ecfp, maccs, mordred_desc, rdkit_desc


def compute_descriptor_vector(smiles: str, descriptor_name: str) -> np.ndarray:
    if descriptor_name == "RDKit":
        return np.asarray(rdkit_desc(smiles), dtype=float)
    if descriptor_name == "ECFP_r1":
        return np.asarray(ecfp(smiles, radius=1), dtype=float)
    if descriptor_name == "ECFP_r2":
        return np.asarray(ecfp(smiles, radius=2), dtype=float)
    if descriptor_name == "ECFP_r3":
        return np.asarray(ecfp(smiles, radius=3), dtype=float)
    if descriptor_name == "MACCS":
        return np.asarray(maccs(smiles), dtype=float)
    if descriptor_name == "Mordred":
        return np.asarray(mordred_desc(smiles), dtype=float)
    raise ValueError(f"Unsupported descriptor: {descriptor_name}")


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _entry_key(entry: Dict) -> tuple[str, str]:
    return str(entry.get("descriptor_name", "")), str(entry.get("model_name", type(entry.get("model")).__name__))


def best_model_entry(bundle: Dict) -> Dict:
    return {
        "descriptor_name": bundle["descriptor_name"],
        "model_name": bundle.get("model_name", type(bundle["model"]).__name__),
        "imputer": bundle["imputer"],
        "scaler": bundle.get("scaler"),
        "model": bundle["model"],
        "r2": float(bundle.get("best_r2", np.nan)),
    }


def ensemble_model_entries(bundle: Dict, max_models: int = 5, positive_only: bool = True) -> List[Dict]:
    entries: List[Dict] = [best_model_entry(bundle)]
    extras = bundle.get("ensemble_models") or bundle.get("uncertainty_models") or []
    for entry in extras:
        entries.append(
            {
                "descriptor_name": entry["descriptor_name"],
                "model_name": entry.get("model_name", type(entry["model"]).__name__),
                "imputer": entry["imputer"],
                "scaler": entry.get("scaler"),
                "model": entry["model"],
                "r2": float(entry.get("r2", np.nan)),
            }
        )

    deduped: List[Dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = _entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    if positive_only:
        positive = [e for e in deduped if np.isnan(e.get("r2", np.nan)) or float(e.get("r2", 0.0)) > 0.0]
        if positive:
            deduped = positive

    deduped.sort(key=lambda e: float(e.get("r2", -1e9)), reverse=True)

    best_key = _entry_key(best_model_entry(bundle))
    best = next((e for e in deduped if _entry_key(e) == best_key), None)
    remainder = [e for e in deduped if _entry_key(e) != best_key]
    ordered = ([best] if best is not None else []) + remainder
    if not ordered:
        ordered = [best_model_entry(bundle)]
    return ordered[: max(1, max_models)]


def predict_with_entries(smiles_list: List[str], entries: Iterable[Dict]) -> Dict[str, np.ndarray]:
    entries = list(entries)
    n = len(smiles_list)
    valid_mask = np.array([Chem.MolFromSmiles(str(smi)) is not None for smi in smiles_list], dtype=bool)
    valid_smiles = [str(smi) for smi, ok in zip(smiles_list, valid_mask) if ok]

    descriptor_names = sorted({str(e["descriptor_name"]) for e in entries})
    descriptor_mats: Dict[str, np.ndarray] = {}
    for descriptor_name in descriptor_names:
        if valid_smiles:
            mat = np.vstack([compute_descriptor_vector(smi, descriptor_name) for smi in valid_smiles]).astype(float)
            mat[~np.isfinite(mat)] = np.nan
        else:
            mat = np.empty((0, 0), dtype=float)
        descriptor_mats[descriptor_name] = mat

    stacked = np.full((len(entries), n), np.nan, dtype=float)
    model_names: List[str] = []
    descriptor_used: List[str] = []
    r2_values: List[float] = []

    for idx, entry in enumerate(entries):
        descriptor_name = str(entry["descriptor_name"])
        X = descriptor_mats[descriptor_name]
        if X.size == 0:
            model_names.append(str(entry.get("model_name", f"model_{idx+1}")))
            descriptor_used.append(descriptor_name)
            r2_values.append(float(entry.get("r2", np.nan)))
            continue
        Xt = entry["imputer"].transform(X)
        scaler = entry.get("scaler")
        if scaler is not None:
            Xt = scaler.transform(Xt)
        pred = np.asarray(entry["model"].predict(Xt), dtype=float)
        stacked[idx, valid_mask] = pred
        model_names.append(str(entry.get("model_name", f"model_{idx+1}")))
        descriptor_used.append(descriptor_name)
        r2_values.append(float(entry.get("r2", np.nan)))

    return {
        "valid_mask": valid_mask,
        "pred_log_matrix": stacked,
        "model_names": np.asarray(model_names, dtype=object),
        "descriptor_names": np.asarray(descriptor_used, dtype=object),
        "r2_values": np.asarray(r2_values, dtype=float),
    }


def summarize_predictions(smiles_list: List[str], bundle: Dict, max_models: int = 5) -> Dict[str, np.ndarray | int | List[str]]:
    entries = ensemble_model_entries(bundle, max_models=max_models, positive_only=True)
    pred = predict_with_entries(smiles_list, entries)
    mat = pred["pred_log_matrix"]

    if mat.size == 0:
        n = len(smiles_list)
        nan = np.full(n, np.nan, dtype=float)
        return {
            "qsar_model_count": 0,
            "qsar_model_names": [],
            "pred_logIC50_uM": nan.copy(),
            "pred_ic50_uM": nan.copy(),
            "pred_pIC50": nan.copy(),
            "pred_logIC50_uM_ensemble_mean": nan.copy(),
            "pred_logIC50_uM_ensemble_std": nan.copy(),
            "pred_ic50_uM_ensemble_mean": nan.copy(),
            "pred_ic50_uM_ensemble_std": nan.copy(),
            "pred_pIC50_ensemble_mean": nan.copy(),
            "pred_pIC50_ensemble_std": nan.copy(),
            "pred_ic50_uM_lower95": nan.copy(),
            "pred_ic50_uM_upper95": nan.copy(),
            "pred_pIC50_lower95": nan.copy(),
            "pred_pIC50_upper95": nan.copy(),
            "pred_pIC50_uncertainty": nan.copy(),
        }

    with np.errstate(invalid="ignore"):
        best_log = mat[0]
        mean_log = np.nanmean(mat, axis=0)
        std_log = np.nanstd(mat, axis=0, ddof=0)
        uM_matrix = np.power(10.0, mat)
        mean_um = np.nanmean(uM_matrix, axis=0)
        std_um = np.nanstd(uM_matrix, axis=0, ddof=0)
        lower95_log = mean_log - 1.96 * std_log
        upper95_log = mean_log + 1.96 * std_log

    return {
        "qsar_model_count": len(entries),
        "qsar_model_names": [str(x) for x in pred["model_names"].tolist()],
        "pred_logIC50_uM": best_log,
        "pred_ic50_uM": np.power(10.0, best_log),
        "pred_pIC50": 6.0 - best_log,
        "pred_logIC50_uM_ensemble_mean": mean_log,
        "pred_logIC50_uM_ensemble_std": std_log,
        "pred_ic50_uM_ensemble_mean": mean_um,
        "pred_ic50_uM_ensemble_std": std_um,
        "pred_pIC50_ensemble_mean": 6.0 - mean_log,
        "pred_pIC50_ensemble_std": std_log,
        "pred_ic50_uM_lower95": np.power(10.0, lower95_log),
        "pred_ic50_uM_upper95": np.power(10.0, upper95_log),
        "pred_pIC50_lower95": 6.0 - upper95_log,
        "pred_pIC50_upper95": 6.0 - lower95_log,
        "pred_pIC50_uncertainty": std_log,
    }
