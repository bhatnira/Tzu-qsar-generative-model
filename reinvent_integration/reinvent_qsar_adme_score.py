from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_core.descriptors import rdkit_desc
from reinvent_integration.qsar_ensemble import summarize_predictions

_ADMET_MODEL = None
_QIKPROP_EXECUTABLE_CANDIDATES = [
    "/opt/schrodinger/schrodinger2026-1/qikprop",
    "qikprop",
]

_QIKPROP_ALIASES = {
    'pred_qikprop_cLogP': ['QPlogPo/w', 'qikprop_clogp'],
    'pred_qikprop_logS': ['QPlogS', 'qikprop_logs'],
    'pred_qikprop_metab_sites': ['#metab', 'qikprop_metab_sites'],
}


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


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanstd(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=0))


def _qikprop_executable() -> str | None:
    for candidate in _QIKPROP_EXECUTABLE_CANDIDATES:
        if "/" in candidate:
            p = Path(candidate)
            if p.exists() and p.is_file():
                return str(p)
        else:
            from shutil import which
            hit = which(candidate)
            if hit:
                return hit
    return None


def _norm_qikprop_col(value: str) -> str:
    return ''.join(ch for ch in str(value).lower() if ch.isalnum())


def _resolve_qikprop_columns(columns: list[str]) -> dict[str, str]:
    norm_to_col = {_norm_qikprop_col(col): col for col in columns}
    resolved: dict[str, str] = {}
    for target, aliases in _QIKPROP_ALIASES.items():
        for alias in aliases:
            hit = norm_to_col.get(_norm_qikprop_col(alias))
            if hit is not None:
                resolved[target] = hit
                break
    return resolved


def _qikprop_row_index(row: pd.Series) -> int | None:
    raw = row.get('molecule', row.get('title', row.get('_Name', '')))
    if pd.isna(raw):
        return None
    match = re.search(r'(\d+)$', str(raw))
    if not match:
        return None
    try:
        return max(0, int(match.group(1)) - 1)
    except Exception:
        return None


def _read_qikprop_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except ParserError:
        return pd.read_csv(path, engine='python', on_bad_lines='skip')


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
    needed = ['Clearance_Microsome_AZ', 'Clearance_Hepatocyte_AZ', 'Half_Life_Obach', 'logP', 'molecular_weight']
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    def _col_or_nan(name: str) -> list[float]:
        if name not in df.columns:
            return [float('nan')] * len(df)
        return [float(x) if np.isfinite(x) else float('nan') for x in pd.to_numeric(df[name], errors='coerce').to_numpy()]

    return {
        'pred_MetStab_Clearance_Microsome_AZ': [float(x) if not np.isnan(x) else 1e6 for x in df['Clearance_Microsome_AZ'].to_numpy()],
        'pred_MetStab_Clearance_Hepatocyte_AZ': [float(x) if not np.isnan(x) else 1e6 for x in df['Clearance_Hepatocyte_AZ'].to_numpy()],
        'pred_MetStab_Half_Life_Obach': [float(x) if not np.isnan(x) else 0.0 for x in df['Half_Life_Obach'].to_numpy()],
        'pred_admet_logP': _col_or_nan('logP'),
        'pred_admet_molecular_weight': _col_or_nan('molecular_weight'),
        'pred_MetStab_Clearance_Microsome_Human': _col_or_nan('Clearance_Microsome_Human'),
        'pred_MetStab_Clearance_Microsome_Mouse': _col_or_nan('Clearance_Microsome_Mouse'),
        'pred_MetStab_Clearance_Hepatocyte_Human': _col_or_nan('Clearance_Hepatocyte_Human'),
        'pred_MetStab_Clearance_Hepatocyte_Mouse': _col_or_nan('Clearance_Hepatocyte_Mouse'),
        'pred_MetStab_Half_Life_Human': _col_or_nan('Half_Life_Human'),
        'pred_MetStab_Half_Life_Mouse': _col_or_nan('Half_Life_Mouse'),
    }


def _predict_qikprop(smiles_list: List[str]) -> Dict[str, List[float | str]]:
    n = len(smiles_list)
    out = {
        'pred_qikprop_cLogP': [float('nan')] * n,
        'pred_qikprop_logS': [float('nan')] * n,
        'pred_qikprop_metab_sites': [float('nan')] * n,
        'pred_qikprop_status': ['unavailable'] * n,
    }
    executable = _qikprop_executable()
    if executable is None or n == 0:
        return out

    try:
        with tempfile.TemporaryDirectory(prefix='qikprop_reinvent_') as td:
            td = Path(td)
            sdf = td / 'input.sdf'
            writer = Chem.SDWriter(str(sdf))
            kept: list[int] = []
            for idx, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    continue
                mol.SetProp('_Name', f'Molecule_{idx+1:05d}')
                writer.write(mol)
                kept.append(idx)
            writer.close()
            if not kept:
                out['pred_qikprop_status'] = ['no_valid_molecules'] * n
                return out

            subprocess.run([executable, '-WAIT', '-nosim', sdf.name], cwd=td, capture_output=True, text=True, timeout=1800, check=False)
            csv_path = td / 'input.CSV'
            if not csv_path.exists():
                out['pred_qikprop_status'] = ['missing_csv'] * n
                return out

            qdf = _read_qikprop_csv(csv_path)
            resolved_cols = _resolve_qikprop_columns(list(qdf.columns))
            if not resolved_cols:
                out['pred_qikprop_status'] = ['missing_columns'] * n
                return out

            assigned = [False] * n
            processed = [False] * n
            for pos in range(len(qdf)):
                row = qdf.iloc[pos]
                idx = _qikprop_row_index(row)
                if idx is None or idx < 0 or idx >= n:
                    if pos < len(kept):
                        idx = kept[pos]
                    else:
                        continue

                c_col = resolved_cols.get('pred_qikprop_cLogP')
                s_col = resolved_cols.get('pred_qikprop_logS')
                m_col = resolved_cols.get('pred_qikprop_metab_sites')
                c_val = float('nan')
                s_val = float('nan')
                m_val = float('nan')
                if c_col is not None:
                    c_val = float(pd.to_numeric(row.get(c_col, np.nan), errors='coerce'))
                    out['pred_qikprop_cLogP'][idx] = c_val
                if s_col is not None:
                    s_val = float(pd.to_numeric(row.get(s_col, np.nan), errors='coerce'))
                    out['pred_qikprop_logS'][idx] = s_val
                if m_col is not None:
                    m_val = float(pd.to_numeric(row.get(m_col, np.nan), errors='coerce'))
                    out['pred_qikprop_metab_sites'][idx] = m_val
                has_numeric = np.isfinite(c_val) or np.isfinite(s_val) or np.isfinite(m_val)
                out['pred_qikprop_status'][idx] = 'ok' if has_numeric else 'no_prediction'
                assigned[idx] = has_numeric
                processed[idx] = True

            for idx in kept:
                if not processed[idx]:
                    out['pred_qikprop_status'][idx] = 'partial'
            return out
    except Exception:
        out['pred_qikprop_status'] = ['error'] * n
        return out


def _predict_rdkit_rule_proxies(smiles_list: List[str]) -> Dict[str, List[float]]:
    clogp, logs_proxy, metstab_proxy = [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            clogp.append(float('nan'))
            logs_proxy.append(float('nan'))
            metstab_proxy.append(float('nan'))
            continue
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        rot = float(Descriptors.NumRotatableBonds(mol))
        logs_est = -0.012 * mw - 0.55 * logp + 0.003 * tpsa - 1.20
        metstab_est = 70.0 - 7.5 * logp - 1.2 * rot + 0.12 * tpsa
        clogp.append(logp)
        logs_proxy.append(float(logs_est))
        metstab_proxy.append(float(max(1.0, min(120.0, metstab_est))))
    return {
        'pred_rdkit_cLogP': clogp,
        'pred_rdkit_logS_proxy': logs_proxy,
        'pred_rdkit_metstab_proxy': metstab_proxy,
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
    p.add_argument('--metstab-weight', type=float, default=0.40)
    p.add_argument('--uncertainty-penalty', type=float, default=0.15)
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
    qikprop = _predict_qikprop(smiles_list)
    rdkit_proxy = _predict_rdkit_rule_proxies(smiles_list)

    qsar_scores = [_high_better_score(v, args.target_pic50, 0.25) for v in qsar['pred_pic50']]

    clogp_score_sources = {
        'adme_py': [float('nan') if np.isnan(v) else _range_score(v, args.clogp_low, args.clogp_high, 0.35) for v in adme_py['pred_cLogP']],
        'qikprop': [float('nan') if np.isnan(v) else _range_score(v, args.clogp_low, args.clogp_high, 0.35) for v in qikprop['pred_qikprop_cLogP']],
        'admet_ai': [float('nan') if np.isnan(v) else _range_score(v, args.clogp_low, args.clogp_high, 0.35) for v in admet['pred_admet_logP']],
        'rdkit_rule': [float('nan') if np.isnan(v) else _range_score(v, args.clogp_low, args.clogp_high, 0.35) for v in rdkit_proxy['pred_rdkit_cLogP']],
    }

    admet_sol_proxy = []
    for mw, lp in zip(admet['pred_admet_molecular_weight'], admet['pred_admet_logP']):
        if not np.isfinite(mw) or not np.isfinite(lp):
            admet_sol_proxy.append(float('nan'))
        else:
            admet_sol_proxy.append(float(-0.012 * mw - 0.55 * lp - 1.20))

    solubility_score_sources = {
        'adme_py': [float('nan') if np.isnan(v) else _high_better_score(v, args.logs_threshold, 0.35) for v in adme_py['pred_logS_ESOL']],
        'qikprop': [float('nan') if np.isnan(v) else _high_better_score(v, args.logs_threshold, 0.35) for v in qikprop['pred_qikprop_logS']],
        'admet_proxy': [float('nan') if np.isnan(v) else _high_better_score(v, args.logs_threshold, 0.35) for v in admet_sol_proxy],
        'rdkit_rule': [float('nan') if np.isnan(v) else _high_better_score(v, args.logs_threshold, 0.35) for v in rdkit_proxy['pred_rdkit_logS_proxy']],
    }

    microsome_scores = [_low_better_score(v, args.microsome_clearance_max, 10.0) for v in admet['pred_MetStab_Clearance_Microsome_AZ']]
    hepatocyte_scores = [_low_better_score(v, args.hepatocyte_clearance_max, 10.0) for v in admet['pred_MetStab_Clearance_Hepatocyte_AZ']]
    half_life_scores = [_high_better_score(v, args.half_life_min, 8.0) for v in admet['pred_MetStab_Half_Life_Obach']]
    admet_metstab_score = [
        float(0.35 * m + 0.30 * h + 0.35 * hl)
        for m, h, hl in zip(microsome_scores, hepatocyte_scores, half_life_scores)
    ]
    qikprop_metstab_score = [float('nan') if np.isnan(v) else _low_better_score(v, 3.0, 2.0) for v in qikprop['pred_qikprop_metab_sites']]
    rdkit_metstab_score = [float('nan') if np.isnan(v) else _high_better_score(v, 40.0, 20.0) for v in rdkit_proxy['pred_rdkit_metstab_proxy']]
    adme_py_metstab_proxy = [
        float('nan') if (np.isnan(lp) or np.isnan(ls)) else _high_better_score(70.0 - 8.0 * lp + 2.0 * ls, 40.0, 20.0)
        for lp, ls in zip(adme_py['pred_cLogP'], adme_py['pred_logS_ESOL'])
    ]
    metstab_score_sources = {
        'admet_ai': admet_metstab_score,
        'qikprop': qikprop_metstab_score,
        'rdkit_rule': rdkit_metstab_score,
        'adme_py_proxy': adme_py_metstab_proxy,
    }

    disallowed_patterns = _compile_smarts(args.disallow_smarts)
    disallowed = _disallowed_mask(smiles_list, disallowed_patterns)
    penalty = min(max(args.disallow_penalty, 0.0), 1.0)

    clogp_ensemble_score, solubility_ensemble_score, metstab_ensemble_score = [], [], []
    clogp_uncertainty, solubility_uncertainty, metstab_uncertainty = [], [], []

    for i in range(len(smiles_list)):
        cvals = [clogp_score_sources[k][i] for k in clogp_score_sources]
        svals = [solubility_score_sources[k][i] for k in solubility_score_sources]
        mvals = [metstab_score_sources[k][i] for k in metstab_score_sources]
        clogp_ensemble_score.append(_nanmean(cvals))
        solubility_ensemble_score.append(_nanmean(svals))
        metstab_ensemble_score.append(_nanmean(mvals))
        clogp_uncertainty.append(min(1.0, _nanstd(cvals)))
        solubility_uncertainty.append(min(1.0, _nanstd(svals)))
        metstab_uncertainty.append(min(1.0, _nanstd(mvals)))

    total_weight = max(1e-6, args.clogp_weight + args.solubility_weight + args.metstab_weight)
    combined = []
    ensemble_unc = []
    for i in range(len(smiles_list)):
        score = (
            args.clogp_weight * (0.0 if np.isnan(clogp_ensemble_score[i]) else clogp_ensemble_score[i]) +
            args.solubility_weight * (0.0 if np.isnan(solubility_ensemble_score[i]) else solubility_ensemble_score[i]) +
            args.metstab_weight * (0.0 if np.isnan(metstab_ensemble_score[i]) else metstab_ensemble_score[i])
        ) / total_weight
        unc = float((clogp_uncertainty[i] + solubility_uncertainty[i] + metstab_uncertainty[i]) / 3.0)
        score = score * max(0.0, 1.0 - args.uncertainty_penalty * unc)
        if disallowed[i] == 1:
            score = score * (1.0 - penalty)
        combined.append(float(score))
        ensemble_unc.append(float(unc))

    payload = {
        args.property_name: combined,
        'adme3_ensemble_score': [float(x) for x in combined],
        'adme3_ensemble_uncertainty': [float(x) for x in ensemble_unc],
        'pred_log_ic50_uM': [float(x) for x in qsar['pred_log_ic50_uM']],
        'pred_pic50': [float(x) for x in qsar['pred_pic50']],
        'pred_qikprop_status': [str(x) for x in qikprop['pred_qikprop_status']],
        'pred_qikprop_cLogP': [float(x) if not np.isnan(x) else 999.0 for x in qikprop['pred_qikprop_cLogP']],
        'pred_qikprop_logS': [float(x) if not np.isnan(x) else -999.0 for x in qikprop['pred_qikprop_logS']],
        'pred_qikprop_metab_sites': [float(x) if not np.isnan(x) else 999.0 for x in qikprop['pred_qikprop_metab_sites']],
        'pred_rdkit_cLogP': [float(x) if not np.isnan(x) else 999.0 for x in rdkit_proxy['pred_rdkit_cLogP']],
        'pred_rdkit_logS_proxy': [float(x) if not np.isnan(x) else -999.0 for x in rdkit_proxy['pred_rdkit_logS_proxy']],
        'pred_rdkit_metstab_proxy': [float(x) if not np.isnan(x) else 0.0 for x in rdkit_proxy['pred_rdkit_metstab_proxy']],
        'pred_admet_logP': [float(x) if not np.isnan(x) else 999.0 for x in admet['pred_admet_logP']],
        'pred_cLogP': [float(x) if not np.isnan(x) else 999.0 for x in adme_py['pred_cLogP']],
        'pred_logS_ESOL': [float(x) if not np.isnan(x) else -999.0 for x in adme_py['pred_logS_ESOL']],
        'pred_Solubility_ESOL_mol_per_L': [float(x) if not np.isnan(x) else 0.0 for x in adme_py['pred_Solubility_ESOL_mol_per_L']],
        'pred_MetStab_Clearance_Microsome_AZ': [float(x) for x in admet['pred_MetStab_Clearance_Microsome_AZ']],
        'pred_MetStab_Clearance_Hepatocyte_AZ': [float(x) for x in admet['pred_MetStab_Clearance_Hepatocyte_AZ']],
        'pred_MetStab_Half_Life_Obach': [float(x) for x in admet['pred_MetStab_Half_Life_Obach']],
        'pred_pic50_uncertainty': [float(x) for x in qsar['pred_pic50_uncertainty']],
        'qsar_score': [float(x) for x in qsar_scores],
        'clogp_ensemble_score': [float(0.0 if np.isnan(x) else x) for x in clogp_ensemble_score],
        'solubility_ensemble_score': [float(0.0 if np.isnan(x) else x) for x in solubility_ensemble_score],
        'metstab_ensemble_score': [float(0.0 if np.isnan(x) else x) for x in metstab_ensemble_score],
        'clogp_uncertainty': [float(x) for x in clogp_uncertainty],
        'solubility_uncertainty': [float(x) for x in solubility_uncertainty],
        'metstab_uncertainty': [float(x) for x in metstab_uncertainty],
        'microsome_score': [float(x) for x in microsome_scores],
        'hepatocyte_score': [float(x) for x in hepatocyte_scores],
        'half_life_score': [float(x) for x in half_life_scores],
        'pred_MetStab_Clearance_Microsome_Human': [float(x) if not np.isnan(x) else 1e6 for x in admet['pred_MetStab_Clearance_Microsome_Human']],
        'pred_MetStab_Clearance_Microsome_Mouse': [float(x) if not np.isnan(x) else 1e6 for x in admet['pred_MetStab_Clearance_Microsome_Mouse']],
        'pred_MetStab_Clearance_Hepatocyte_Human': [float(x) if not np.isnan(x) else 1e6 for x in admet['pred_MetStab_Clearance_Hepatocyte_Human']],
        'pred_MetStab_Clearance_Hepatocyte_Mouse': [float(x) if not np.isnan(x) else 1e6 for x in admet['pred_MetStab_Clearance_Hepatocyte_Mouse']],
        'pred_MetStab_Half_Life_Human': [float(x) if not np.isnan(x) else 0.0 for x in admet['pred_MetStab_Half_Life_Human']],
        'pred_MetStab_Half_Life_Mouse': [float(x) if not np.isnan(x) else 0.0 for x in admet['pred_MetStab_Half_Life_Mouse']],
        'disallowed_smarts': [int(x) for x in disallowed],
    }
    print(json.dumps({'version': 1, 'payload': payload}))


if __name__ == '__main__':
    main()
