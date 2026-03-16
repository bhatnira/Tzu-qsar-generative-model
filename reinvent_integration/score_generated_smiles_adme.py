from __future__ import annotations

import argparse
import contextlib
import io
from importlib import import_module
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import xlsxwriter
from adme_py import ADME
from admet_ai import ADMETModel
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Draw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_core.descriptors import rdkit_desc
from reinvent_integration.qsar_ensemble import summarize_predictions

RDLogger.DisableLog('rdApp.*')

IMG_W, IMG_H = 170, 120
ROW_H_PT = 92
IMG_COL_W = 24

CUTS = {
    'pred_ic50_uM_max': 5.0,
    'pred_cLogP_min': 1.0,
    'pred_cLogP_max': 3.5,
    'pred_logS_ESOL_min': -6.0,
    'pred_MetStab_Clearance_Microsome_AZ_max': 70.0,
    'pred_MetStab_Clearance_Hepatocyte_AZ_max': 80.0,
    'pred_MetStab_Half_Life_Obach_min': 40.0,
}

TARGET_FREE_ADMET_COLUMNS = [
    'PAINS_alert', 'BRENK_alert', 'NIH_alert',
    'AMES', 'ClinTox', 'DILI', 'Carcinogens_Lagunin', 'hERG',
    'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
    'HIA_Hou', 'Bioavailability_Ma', 'PAMPA_NCATS', 'Pgp_Broccatelli',
    'Caco2_Wang', 'PPBR_AZ', 'VDss_Lombardo',
    'molecular_weight', 'logP', 'QED', 'Lipinski'
]


def mol_png(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(IMG_W, IMG_H))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def write_xlsx(df: pd.DataFrame, path: Path, smiles_col: str = 'canonical_smiles') -> None:
    wb = xlsxwriter.Workbook(str(path), {'constant_memory': True})
    ws = wb.add_worksheet('Compounds')
    hdr = wb.add_format({'bold': True, 'bg_color': '#D0E4F7', 'align': 'center', 'valign': 'vcenter', 'border': 1})
    txt = wb.add_format({'valign': 'vcenter', 'border': 1})
    num = wb.add_format({'valign': 'vcenter', 'num_format': '0.0000', 'border': 1})
    ws.write(0, 0, 'Structure', hdr)
    for c, col in enumerate(df.columns, start=1):
        ws.write(0, c, col, hdr)
        ws.set_column(c, c, min(max(len(col) + 2, 12), 44))
    ws.set_column(0, 0, IMG_COL_W)
    for r, (_, row) in enumerate(df.iterrows(), start=1):
        ws.set_row(r, ROW_H_PT)
        buf = mol_png(row.get(smiles_col, ''))
        if buf is not None:
            ws.insert_image(r, 0, f'mol_{r}.png', {'image_data': buf, 'x_offset': 2, 'y_offset': 2, 'positioning': 1})
        for c, val in enumerate(row.values, start=1):
            try:
                ws.write_number(r, c, float(val), num)
            except Exception:
                ws.write(r, c, '' if val is None else str(val), txt)
    wb.close()


def load_generated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'canonical_smiles'
    input_col = 'Input_SMILES' if 'Input_SMILES' in df.columns else None
    nll_col = 'NLL' if 'NLL' in df.columns else None
    state_col = 'SMILES_state' if 'SMILES_state' in df.columns else None
    tanimoto_col = 'Tanimoto' if 'Tanimoto' in df.columns else None

    rows = []
    for _, row in df.iterrows():
        smi = str(row.get(smiles_col, '')).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if len(can) < 8:
            continue
        entry = {'canonical_smiles': can}
        if input_col:
            entry['Input_SMILES'] = row.get(input_col)
        if state_col:
            entry['SMILES_state'] = row.get(state_col)
        if tanimoto_col:
            entry['Tanimoto'] = row.get(tanimoto_col)
        if nll_col:
            entry['NLL'] = row.get(nll_col)
        rows.append(entry)

    dedup_cols = ['canonical_smiles']
    return pd.DataFrame(rows).drop_duplicates(subset=dedup_cols).copy()


def filter_excluded_smarts(df: pd.DataFrame, exclude_smarts: list[str]) -> pd.DataFrame:
    if df.empty or not exclude_smarts:
        return df
    patterns = []
    for smarts in exclude_smarts:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            patterns.append(patt)
    if not patterns:
        return df

    keep = []
    for smi in df['canonical_smiles'].tolist():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            keep.append(False)
            continue
        is_excluded = any(mol.HasSubstructMatch(patt) for patt in patterns)
        keep.append(not is_excluded)
    return df.loc[keep].reset_index(drop=True)


def filter_max_nitro_groups(df: pd.DataFrame, max_nitro_groups: int | None) -> pd.DataFrame:
    if df.empty or max_nitro_groups is None:
        return df

    patt_charged = Chem.MolFromSmarts('[N+](=O)[O-]')
    patt_neutral = Chem.MolFromSmarts('N(=O)=O')

    if patt_charged is None and patt_neutral is None:
        return df

    keep = []
    for smi in df['canonical_smiles'].tolist():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            keep.append(False)
            continue

        nitro_count = 0
        if patt_charged is not None:
            nitro_count += len(mol.GetSubstructMatches(patt_charged))
        if patt_neutral is not None:
            nitro_count += len(mol.GetSubstructMatches(patt_neutral))

        keep.append(nitro_count <= max_nitro_groups)

    return df.loc[keep].reset_index(drop=True)


def _sa_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return float('nan')
    try:
        sascorer = import_module('rdkit.Contrib.SA_Score.sascorer')
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float('nan')


def score_df(df: pd.DataFrame, qsar_model_path: Path) -> pd.DataFrame:
    bundle = joblib.load(qsar_model_path)
    admet = ADMETModel()
    smiles_list = df['canonical_smiles'].astype(str).tolist()

    qsar_summary = summarize_predictions(smiles_list, bundle, max_models=5)
    for col, values in qsar_summary.items():
        if col == 'qsar_model_names':
            df['qsar_models_used'] = ' | '.join(values)
        elif np.isscalar(values):
            df[col] = values
        else:
            df[col] = np.asarray(values)

    if 'qsar_model_count' in qsar_summary:
        df['qsar_model_count'] = int(qsar_summary['qsar_model_count'])

    adme_rows = []
    for s in df['canonical_smiles']:
        props = ADME(s).properties
        adme_rows.append({
            'pred_cLogP': props['lipophilicity'].get('wlogp', np.nan),
            'pred_logS_ESOL': props['solubility'].get('log_s_esol', np.nan),
            'pred_Solubility_ESOL_mol_per_L': props['solubility'].get('solubility_esol', np.nan),
            'pred_Solubility_ESOL_class': props['solubility'].get('class_esol', None),
        })
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(adme_rows)], axis=1)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        admet_df = admet.predict(smiles=smiles_list).reset_index(drop=True)
    df['pred_MetStab_Clearance_Microsome_AZ'] = admet_df['Clearance_Microsome_AZ']
    df['pred_MetStab_Clearance_Hepatocyte_AZ'] = admet_df['Clearance_Hepatocyte_AZ']
    df['pred_MetStab_Half_Life_Obach'] = admet_df['Half_Life_Obach']
    for col in TARGET_FREE_ADMET_COLUMNS:
        if col in admet_df.columns and col not in df.columns:
            df[col] = admet_df[col]

    df['sa_score'] = df['canonical_smiles'].map(_sa_score)
    df['sa_accessibility'] = 1.0 - ((pd.to_numeric(df['sa_score'], errors='coerce') - 1.0) / 9.0).clip(lower=0.0, upper=1.0)
    alert_cols = [c for c in ['PAINS_alert', 'BRENK_alert', 'NIH_alert'] if c in df.columns]
    if alert_cols:
        df['liability_alert_count'] = df[alert_cols].fillna(0).sum(axis=1)
    else:
        df['liability_alert_count'] = 0.0
    df['passes_qsar_uncertainty'] = (pd.to_numeric(df['pred_ic50_uM_upper95'], errors='coerce') <= CUTS['pred_ic50_uM_max']).astype(int)

    df['passes_optimal'] = (
        (df['pred_ic50_uM'] <= CUTS['pred_ic50_uM_max'])
        & (df['pred_cLogP'] >= CUTS['pred_cLogP_min'])
        & (df['pred_cLogP'] <= CUTS['pred_cLogP_max'])
        & (df['pred_logS_ESOL'] >= CUTS['pred_logS_ESOL_min'])
        & (df['pred_MetStab_Clearance_Microsome_AZ'] <= CUTS['pred_MetStab_Clearance_Microsome_AZ_max'])
        & (df['pred_MetStab_Clearance_Hepatocyte_AZ'] <= CUTS['pred_MetStab_Clearance_Hepatocyte_AZ_max'])
        & (df['pred_MetStab_Half_Life_Obach'] >= CUTS['pred_MetStab_Half_Life_Obach_min'])
    ).astype(int)

    return df.sort_values(
        ['passes_optimal', 'passes_qsar_uncertainty', 'pred_ic50_uM', 'pred_pIC50_uncertainty'],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def export_results(df: pd.DataFrame, outdir: Path, label: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    hits = df[df['passes_optimal'] == 1].copy()
    all_csv = outdir / f'{label}_generated_scored.csv'
    hits_csv = outdir / f'{label}_generated_optimal.csv'
    all_xlsx = outdir / f'{label}_generated_scored.xlsx'
    hits_xlsx = outdir / f'{label}_generated_optimal.xlsx'
    manifest = outdir / 'manifest.csv'

    df.to_csv(all_csv, index=False)
    hits.to_csv(hits_csv, index=False)
    write_xlsx(df, all_xlsx)
    write_xlsx(hits, hits_xlsx)
    pd.DataFrame([
        {'file': all_csv.name, 'rows': len(df)},
        {'file': hits_csv.name, 'rows': len(hits)},
        {'file': all_xlsx.name, 'rows': len(df)},
        {'file': hits_xlsx.name, 'rows': len(hits)},
        {'cutoff': 'IC50<=5uM; cLogP 1-3.5; logS>=-6.0; Microsome<=70; Hepatocyte<=80; HalfLife>=40; QSAR uncertainty via up to 5 positive-R2 models'},
    ]).to_csv(manifest, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description='Score generated SMILES with QSAR and ADME, then export CSV/XLSX with structure images.')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--qsar-model', default='reinvent_integration/artifacts/qsar_best_model.joblib')
    parser.add_argument('--exclude-smarts', action='append', default=[])
    parser.add_argument('--max-nitro-groups', type=int, default=None)
    args = parser.parse_args()

    df = load_generated(Path(args.input_csv))
    df = filter_excluded_smarts(df, args.exclude_smarts)
    df = filter_max_nitro_groups(df, args.max_nitro_groups)
    if df.empty:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{'file': f'{args.label}_generated_scored.csv', 'rows': 0}]).to_csv(Path(args.output_dir) / 'manifest.csv', index=False)
        print(f'{args.label} valid_unique=0 optimal=0')
        return 0

    scored = score_df(df, Path(args.qsar_model))
    export_results(scored, Path(args.output_dir), args.label)
    hits = int((scored['passes_optimal'] == 1).sum())
    print(f'{args.label} valid_unique={len(scored)} optimal={hits}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
