from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_core.descriptors import rdkit_desc
from reinvent_integration.qsar_ensemble import summarize_predictions

_ADMET_MODEL = None


def _safe_sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def _range_score(value: float, low: float, high: float, softness: float) -> float:
    return _safe_sigmoid((value - low) / max(1e-6, softness)) * _safe_sigmoid((high - value) / max(1e-6, softness))


def _high_better_score(value: float, threshold: float, softness: float) -> float:
    return _safe_sigmoid((value - threshold) / max(1e-6, softness))


def _low_better_score(value: float, threshold: float, softness: float) -> float:
    return _safe_sigmoid((threshold - value) / max(1e-6, softness))


def _load_admet_model():
    global _ADMET_MODEL
    if _ADMET_MODEL is None:
        from admet_ai import ADMETModel
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _ADMET_MODEL = ADMETModel()
    return _ADMET_MODEL


def _predict_qsar(smiles_list: List[str], model_bundle: Dict) -> Dict[str, List[float]]:
    summary = summarize_predictions(smiles_list, model_bundle, max_models=5)
    pred_log = np.asarray(summary['pred_logIC50_uM'], dtype=float)
    pred_pic50 = np.asarray(summary['pred_pIC50'], dtype=float)
    pred_unc = np.asarray(summary['pred_pIC50_uncertainty'], dtype=float)
    pred_log = np.where(np.isfinite(pred_log), pred_log, 99.0)
    pred_pic50 = np.where(np.isfinite(pred_pic50), pred_pic50, 0.0)
    pred_unc = np.where(np.isfinite(pred_unc), pred_unc, 99.0)
    return {
        'pred_log_ic50_uM': pred_log.tolist(),
        'pred_pic50': pred_pic50.tolist(),
        'pred_pic50_uncertainty': pred_unc.tolist(),
    }


def _predict_adme_py(smiles_list: List[str]) -> Dict[str, List[float | str]]:
    from adme_py import ADME

    clogp, logs, solmol, solclass = [], [], [], []
    for smi in smiles_list:
        try:
            props = ADME(smi).properties
            lip = props.get('lipophilicity', {})
            sol = props.get('solubility', {})
            clogp.append(float(lip.get('wlogp', np.nan)))
            logs.append(float(sol.get('log_s_esol', np.nan)))
            solmol.append(float(sol.get('solubility_esol', np.nan)))
            solclass.append(str(sol.get('class_esol', 'Unknown')))
        except Exception:
            clogp.append(float('nan'))
            logs.append(float('nan'))
            solmol.append(float('nan'))
            solclass.append('Invalid')
    return {
        'pred_cLogP': clogp,
        'pred_logS_ESOL': logs,
        'pred_Solubility_ESOL_mol_per_L': solmol,
        'pred_Solubility_ESOL_class': solclass,
    }


def _predict_admet_ai(smiles_list: List[str]) -> Dict[str, List[float]]:
    model = _load_admet_model()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        df = model.predict(smiles=smiles_list)
    df = df.reset_index(drop=True)
    needed = ['Clearance_Microsome_AZ', 'Clearance_Hepatocyte_AZ', 'Half_Life_Obach']
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    return {
        'pred_MetStab_Clearance_Microsome_AZ': [float(x) if not np.isnan(x) else 1e6 for x in df['Clearance_Microsome_AZ'].to_numpy()],
        'pred_MetStab_Clearance_Hepatocyte_AZ': [float(x) if not np.isnan(x) else 1e6 for x in df['Clearance_Hepatocyte_AZ'].to_numpy()],
        'pred_MetStab_Half_Life_Obach': [float(x) if not np.isnan(x) else 0.0 for x in df['Half_Life_Obach'].to_numpy()],
    }


def _compile_smarts(smarts_list: List[str]) -> List[Chem.Mol]:
    compiled = []
    for smarts in smarts_list:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            compiled.append(patt)
    return compiled


def _disallowed_mask(smiles_list: List[str], patterns: List[Chem.Mol]) -> List[int]:
    if not patterns:
        return [0] * len(smiles_list)
    mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mask.append(1)
            continue
        is_disallowed = any(mol.HasSubstructMatch(patt) for patt in patterns)
        mask.append(1 if is_disallowed else 0)
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--qsar-model', required=True)
    p.add_argument('--target-pic50', type=float, default=5.30103)
    p.add_argument('--clogp-low', type=float, default=1.0)
    p.add_argument('--clogp-high', type=float, default=3.5)
    p.add_argument('--logs-threshold', type=float, default=-6.0)
    p.add_argument('--microsome-clearance-max', type=float, default=70.0)
    p.add_argument('--hepatocyte-clearance-max', type=float, default=80.0)
    p.add_argument('--half-life-min', type=float, default=40.0)
    p.add_argument('--qsar-weight', type=float, default=0.40)
    p.add_argument('--clogp-weight', type=float, default=0.15)
    p.add_argument('--solubility-weight', type=float, default=0.15)
    p.add_argument('--microsome-weight', type=float, default=0.10)
    p.add_argument('--hepatocyte-weight', type=float, default=0.10)
    p.add_argument('--half-life-weight', type=float, default=0.10)
    p.add_argument('--property-name', default='qsar_adme_score')
    p.add_argument('--disallow-smarts', action='append', default=[])
    p.add_argument('--disallow-penalty', type=float, default=1.0)
    args = p.parse_args()

    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    if not smiles_list:
        payload = {args.property_name: []}
        print(json.dumps({'version': 1, 'payload': payload}))
        return

    model_bundle = joblib.load(args.qsar_model)
    qsar = _predict_qsar(smiles_list, model_bundle)
    adme_py = _predict_adme_py(smiles_list)
    admet = _predict_admet_ai(smiles_list)

    qsar_scores = [_high_better_score(v, args.target_pic50, 0.25) for v in qsar['pred_pic50']]
    clogp_scores = [0.0 if np.isnan(v) else _range_score(v, args.clogp_low, args.clogp_high, 0.35) for v in adme_py['pred_cLogP']]
    sol_scores = [0.0 if np.isnan(v) else _high_better_score(v, args.logs_threshold, 0.35) for v in adme_py['pred_logS_ESOL']]
    microsome_scores = [_low_better_score(v, args.microsome_clearance_max, 10.0) for v in admet['pred_MetStab_Clearance_Microsome_AZ']]
    hepatocyte_scores = [_low_better_score(v, args.hepatocyte_clearance_max, 10.0) for v in admet['pred_MetStab_Clearance_Hepatocyte_AZ']]
    half_life_scores = [_high_better_score(v, args.half_life_min, 8.0) for v in admet['pred_MetStab_Half_Life_Obach']]

    total_weight = max(1e-6, args.qsar_weight + args.clogp_weight + args.solubility_weight + args.microsome_weight + args.hepatocyte_weight + args.half_life_weight)
    disallowed_patterns = _compile_smarts(args.disallow_smarts)
    disallowed = _disallowed_mask(smiles_list, disallowed_patterns)
    penalty = min(max(args.disallow_penalty, 0.0), 1.0)
    combined = []
    for i in range(len(smiles_list)):
        score = (
            args.qsar_weight * qsar_scores[i] +
            args.clogp_weight * clogp_scores[i] +
            args.solubility_weight * sol_scores[i] +
            args.microsome_weight * microsome_scores[i] +
            args.hepatocyte_weight * hepatocyte_scores[i] +
            args.half_life_weight * half_life_scores[i]
        ) / total_weight
        if disallowed[i] == 1:
            score = score * (1.0 - penalty)
        combined.append(float(score))

    payload = {
        args.property_name: combined,
        'pred_log_ic50_uM': [float(x) for x in qsar['pred_log_ic50_uM']],
        'pred_pic50': [float(x) for x in qsar['pred_pic50']],
        'pred_cLogP': [float(x) if not np.isnan(x) else 999.0 for x in adme_py['pred_cLogP']],
        'pred_logS_ESOL': [float(x) if not np.isnan(x) else -999.0 for x in adme_py['pred_logS_ESOL']],
        'pred_Solubility_ESOL_mol_per_L': [float(x) if not np.isnan(x) else 0.0 for x in adme_py['pred_Solubility_ESOL_mol_per_L']],
        'pred_MetStab_Clearance_Microsome_AZ': [float(x) for x in admet['pred_MetStab_Clearance_Microsome_AZ']],
        'pred_MetStab_Clearance_Hepatocyte_AZ': [float(x) for x in admet['pred_MetStab_Clearance_Hepatocyte_AZ']],
        'pred_MetStab_Half_Life_Obach': [float(x) for x in admet['pred_MetStab_Half_Life_Obach']],
        'pred_pic50_uncertainty': [float(x) for x in qsar['pred_pic50_uncertainty']],
        'qsar_score': [float(x) for x in qsar_scores],
        'clogp_score': [float(x) for x in clogp_scores],
        'solubility_score': [float(x) for x in sol_scores],
        'microsome_score': [float(x) for x in microsome_scores],
        'hepatocyte_score': [float(x) for x in hepatocyte_scores],
        'half_life_score': [float(x) for x in half_life_scores],
        'disallowed_smarts': [int(x) for x in disallowed],
    }
    print(json.dumps({'version': 1, 'payload': payload}))


if __name__ == '__main__':
    main()
