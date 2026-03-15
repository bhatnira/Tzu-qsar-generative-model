#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
from importlib import import_module
from pathlib import Path
import sys
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import xlsxwriter
from admet_ai import ADMETModel
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdShapeHelpers
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reinvent_integration.qsar_ensemble import summarize_predictions


IMG_W, IMG_H = 180, 130
ROW_H_PT = 98
IMG_COL_W = 26

QSAR_MODEL_PATH = Path(__file__).resolve().parents[1] / "reinvent_integration" / "artifacts" / "qsar_best_model.joblib"
REFERENCE_LABELED = Path(__file__).resolve().parents[1] / "reinvent_integration" / "data" / "series_ce_labeled.csv"
REFERENCE_SMI = Path(__file__).resolve().parents[1] / "reinvent_integration" / "data" / "series_ce_unique.smi"
TARGET_FREE_ADMET_COLUMNS = [
    "PAINS_alert", "BRENK_alert", "NIH_alert",
    "AMES", "ClinTox", "DILI", "Carcinogens_Lagunin", "hERG",
    "CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
    "HIA_Hou", "Bioavailability_Ma", "PAMPA_NCATS", "Pgp_Broccatelli",
    "Caco2_Wang", "PPBR_AZ", "VDss_Lombardo",
]

_GREEN_COLS = {"passes_optimal", "pred_pIC50", "pred_MetStab_Half_Life_Obach", "triage_score", "gate2_target_free_score", "gate3_translational_score", "experimental_priority_score", "uncertainty_confidence_score", "liability_free"}
_RED_COLS   = {"pred_ic50_uM", "pred_MetStab_Clearance_Microsome_AZ", "pred_MetStab_Clearance_Hepatocyte_AZ", "pred_pIC50_uncertainty", "hERG", "DILI", "ClinTox", "AMES", "max_cyp_inhibition"}

_ADME_CUTS = {
    "pred_ic50_uM":                           ("<=", 5.0),
    "pred_cLogP":                             ("1-3.5", None),
    "pred_logS_ESOL":                         (">=", -6.0),
    "pred_MetStab_Clearance_Microsome_AZ":    ("<=", 70.0),
    "pred_MetStab_Clearance_Hepatocyte_AZ":   ("<=", 80.0),
    "pred_MetStab_Half_Life_Obach":           (">=", 40.0),
}


def _mol_weight(smi: str) -> float:
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(str(smi))
    return round(Descriptors.ExactMolWt(mol), 2) if mol else float("nan")


def _passes_adme(row: pd.Series) -> int:
    try:
        ic50  = float(row.get("pred_ic50_uM", float("nan")))
        clogp = float(row.get("pred_cLogP",  float("nan")))
        logs  = float(row.get("pred_logS_ESOL", float("nan")))
        mic   = float(row.get("pred_MetStab_Clearance_Microsome_AZ", float("nan")))
        hep   = float(row.get("pred_MetStab_Clearance_Hepatocyte_AZ", float("nan")))
        hl    = float(row.get("pred_MetStab_Half_Life_Obach", float("nan")))
        return int(
            ic50  <= 5.0
            and 1.0 <= clogp <= 3.5
            and logs  >= -6.0
            and mic   <= 70.0
            and hep   <= 80.0
            and hl    >= 40.0
        )
    except Exception:
        return 0


def _enrich(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    """Add MW and ADME pass columns if not already present."""
    out = df.copy()
    if "mol_weight" not in out.columns:
        out["mol_weight"] = out[smiles_col].map(_mol_weight)
    if "passes_adme" not in out.columns:
        out["passes_adme"] = out.apply(_passes_adme, axis=1)
    return out


def _add_compounds_sheet(
    wb: xlsxwriter.Workbook,
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    sheet_name: str = "Compounds",
) -> None:
    ws = wb.add_worksheet(sheet_name)
    hdr  = wb.add_format({"bold": True, "bg_color": "#D0E4F7", "align": "center", "valign": "vcenter", "border": 1, "font_size": 10})
    txt  = wb.add_format({"valign": "vcenter", "border": 1, "text_wrap": True, "font_size": 9})
    num  = wb.add_format({"valign": "vcenter", "num_format": "0.000",  "border": 1, "font_size": 9})
    num2 = wb.add_format({"valign": "vcenter", "num_format": "0.00",   "border": 1, "font_size": 9})
    hi   = wb.add_format({"valign": "vcenter", "num_format": "0.000",  "border": 1, "bg_color": "#C6EFCE", "font_size": 9})
    lo   = wb.add_format({"valign": "vcenter", "num_format": "0.000",  "border": 1, "bg_color": "#FFC7CE", "font_size": 9})
    int_ = wb.add_format({"valign": "vcenter", "num_format": "0",       "border": 1, "font_size": 9})
    pass_fmt  = wb.add_format({"valign": "vcenter", "num_format": "0", "border": 1, "bg_color": "#C6EFCE", "bold": True, "font_size": 9})
    fail_fmt  = wb.add_format({"valign": "vcenter", "num_format": "0", "border": 1, "bg_color": "#FFC7CE", "bold": True, "font_size": 9})

    ws.freeze_panes(1, 0)
    ws.write(0, 0, "Structure", hdr)
    col_names = list(df.columns)
    for c, col in enumerate(col_names, start=1):
        ws.write(0, c, col, hdr)
        ws.set_column(c, c, min(max(len(str(col)) + 2, 12), 52))
    ws.set_column(0, 0, IMG_COL_W)

    for r, (_, row) in enumerate(df.iterrows(), start=1):
        ws.set_row(r, ROW_H_PT)
        smi = str(row.get(smiles_col, ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                AllChem.Compute2DCoords(mol)
                img = Draw.MolToImage(mol, size=(IMG_W, IMG_H))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                ws.insert_image(r, 0, f"mol_{r}.png", {"image_data": buf, "x_offset": 2, "y_offset": 2, "positioning": 1})
            except Exception:
                ws.write(r, 0, smi, txt)
        else:
            ws.write(r, 0, smi, txt)

        for c, val in enumerate(row.values, start=1):
            cname = col_names[c - 1]
            try:
                fval = float(val)
                if cname == "passes_adme":
                    ws.write_number(r, c, int(fval), pass_fmt if int(fval) == 1 else fail_fmt)
                elif cname in {"rank", "mol_weight"}:
                    ws.write_number(r, c, fval, num2)
                elif cname in _GREEN_COLS:
                    ws.write_number(r, c, fval, hi)
                elif cname in _RED_COLS:
                    ws.write_number(r, c, fval, lo)
                else:
                    ws.write_number(r, c, fval, num)
            except Exception:
                ws.write(r, c, "" if val is None else str(val), txt)


def _add_analysis_sheet(wb: xlsxwriter.Workbook, df: pd.DataFrame) -> None:
    ws = wb.add_worksheet("Analysis")
    title  = wb.add_format({"bold": True, "font_size": 13, "font_color": "#1F497D", "bottom": 2})
    sec    = wb.add_format({"bold": True, "font_size": 11, "bg_color": "#D0E4F7", "border": 1})
    hdr    = wb.add_format({"bold": True, "bg_color": "#E2EFDA", "border": 1, "font_size": 9, "align": "center"})
    lbl    = wb.add_format({"bold": True, "border": 1, "font_size": 9})
    val    = wb.add_format({"border": 1, "font_size": 9, "num_format": "0.000"})
    val2   = wb.add_format({"border": 1, "font_size": 9, "num_format": "0.00"})
    val_i  = wb.add_format({"border": 1, "font_size": 9, "num_format": "0"})
    txt_c  = wb.add_format({"border": 1, "font_size": 9})
    pct    = wb.add_format({"border": 1, "font_size": 9, "num_format": "0.0%"})
    hi_v   = wb.add_format({"border": 1, "font_size": 9, "num_format": "0.000", "bg_color": "#C6EFCE"})
    lo_v   = wb.add_format({"border": 1, "font_size": 9, "num_format": "0.000", "bg_color": "#FFC7CE"})

    ws.set_column(0, 0, 38)
    ws.set_column(1, 6, 16)
    ws.freeze_panes(0, 0)

    row = 0
    def _blank(n=1):
        nonlocal row
        row += n

    # ── Title ────────────────────────────────────────────────────────────────
    ws.merge_range(row, 0, row, 5, "HYBRID SHORTLIST — DETAILED ANALYSIS", title)
    _blank(2)

    # ── Section 1: Overall summary stats ─────────────────────────────────────
    ws.merge_range(row, 0, row, 5, "1. OVERALL SUMMARY STATISTICS", sec)
    _blank()
    metric_cols = [
        ("n_compounds",                          "Count",             None),
        ("passes_adme",                          "Passes all ADME criteria (count)", None),
        ("triage_score",                         "Triage score (mean ± std)",         "mean_std"),
        ("pred_pIC50",                           "pred pIC50 (mean ± std)",            "mean_std"),
        ("pred_ic50_uM",                         "pred IC50 µM (median)",              "median"),
        ("pred_cLogP",                           "cLogP (mean ± std)",                 "mean_std"),
        ("pred_logS_ESOL",                       "logS ESOL (mean ± std)",             "mean_std"),
        ("mol_weight",                           "MW (mean ± std)",                   "mean_std"),
        ("pred_MetStab_Clearance_Microsome_AZ",  "Microsome clearance (median)",       "median"),
        ("pred_MetStab_Clearance_Hepatocyte_AZ", "Hepatocyte clearance (median)",      "median"),
        ("pred_MetStab_Half_Life_Obach",         "Half-life Obach (median)",           "median"),
        ("scaffold_bucket",                      "Unique Murcko scaffolds",            "nunique"),
    ]
    ws.write(row, 0, "Metric", hdr)
    ws.write(row, 1, "Value", hdr)
    ws.write(row, 2, "Std dev", hdr)
    ws.write(row, 3, "ADME cutoff", hdr)
    _blank()
    for col_key, label, agg in metric_cols:
        ws.write(row, 0, label, lbl)
        cut_str = ""
        cur_val = val
        if col_key == "n_compounds":
            ws.write_number(row, 1, len(df), val_i)
            ws.write(row, 2, "", txt_c)
        elif col_key == "passes_adme" and "passes_adme" in df.columns:
            n_pass = int(df["passes_adme"].sum())
            ws.write_number(row, 1, n_pass, val_i)
            ws.write(row, 2, f"{n_pass/max(len(df),1)*100:.1f}%", txt_c)
            cut_str = "all criteria"
        elif col_key not in df.columns:
            ws.write(row, 1, "n/a", txt_c)
            ws.write(row, 2, "", txt_c)
        elif agg == "mean_std":
            s = pd.to_numeric(df[col_key], errors="coerce")
            ws.write_number(row, 1, round(float(s.mean()), 4), val)
            ws.write_number(row, 2, round(float(s.std()), 4), val)
            if col_key in _ADME_CUTS:
                op, thresh = _ADME_CUTS[col_key]
                cut_str = f"{op} {thresh}" if thresh is not None else str(op)
        elif agg == "median":
            s = pd.to_numeric(df[col_key], errors="coerce")
            ws.write_number(row, 1, round(float(s.median()), 4),
                            hi_v if col_key in _GREEN_COLS else (lo_v if col_key in _RED_COLS else val))
            ws.write(row, 2, "", txt_c)
            if col_key in _ADME_CUTS:
                op, thresh = _ADME_CUTS[col_key]
                cut_str = f"{op} {thresh}" if thresh is not None else str(op)
        elif agg == "nunique":
            ws.write_number(row, 1, int(df[col_key].nunique()), val_i)
            ws.write(row, 2, "", txt_c)
        ws.write(row, 3, cut_str, txt_c)
        _blank()
    _blank()

    # ── Section 2: Per source family ─────────────────────────────────────────
    ws.merge_range(row, 0, row, 5, "2. CANDIDATES BY SOURCE FAMILY", sec)
    _blank()
    ws.write(row, 0, "Source family",   hdr)
    ws.write(row, 1, "Count",           hdr)
    ws.write(row, 2, "% of shortlist",  hdr)
    ws.write(row, 3, "Mean triage score", hdr)
    ws.write(row, 4, "Mean pIC50",      hdr)
    ws.write(row, 5, "Passes ADME (%)", hdr)
    _blank()
    if "source_family" in df.columns:
        for fam, grp in df.groupby("source_family", sort=False):
            ws.write(row, 0, str(fam), txt_c)
            ws.write_number(row, 1, len(grp), val_i)
            ws.write_number(row, 2, len(grp) / max(len(df), 1), pct)
            ts = pd.to_numeric(grp.get("triage_score", pd.Series(dtype=float)), errors="coerce")
            ws.write_number(row, 3, round(float(ts.mean()), 4), val)
            pi = pd.to_numeric(grp.get("pred_pIC50", pd.Series(dtype=float)), errors="coerce")
            ws.write_number(row, 4, round(float(pi.mean()), 4), val)
            if "passes_adme" in grp.columns:
                ws.write_number(row, 5, grp["passes_adme"].mean(), pct)
            else:
                ws.write(row, 5, "", txt_c)
            _blank()
    _blank()

    # ── Section 3: Per generation mode ───────────────────────────────────────
    ws.merge_range(row, 0, row, 5, "3. CANDIDATES BY GENERATION MODE", sec)
    _blank()
    ws.write(row, 0, "Mode",              hdr)
    ws.write(row, 1, "Count",             hdr)
    ws.write(row, 2, "% of shortlist",    hdr)
    ws.write(row, 3, "Mean triage score", hdr)
    ws.write(row, 4, "Mean pIC50",        hdr)
    ws.write(row, 5, "Passes ADME (%)",   hdr)
    _blank()
    if "source_mode" in df.columns:
        for mode, grp in df.groupby("source_mode", sort=False):
            ws.write(row, 0, str(mode), txt_c)
            ws.write_number(row, 1, len(grp), val_i)
            ws.write_number(row, 2, len(grp) / max(len(df), 1), pct)
            ts = pd.to_numeric(grp.get("triage_score", pd.Series(dtype=float)), errors="coerce")
            ws.write_number(row, 3, round(float(ts.mean()), 4), val)
            pi = pd.to_numeric(grp.get("pred_pIC50", pd.Series(dtype=float)), errors="coerce")
            ws.write_number(row, 4, round(float(pi.mean()), 4), val)
            if "passes_adme" in grp.columns:
                ws.write_number(row, 5, grp["passes_adme"].mean(), pct)
            else:
                ws.write(row, 5, "", txt_c)
            _blank()
    _blank()

    # ── Section 4: Top 15 scaffolds ───────────────────────────────────────────
    ws.merge_range(row, 0, row, 5, "4. TOP 15 MURCKO SCAFFOLD CLUSTERS", sec)
    _blank()
    ws.write(row, 0, "Scaffold SMILES",   hdr)
    ws.write(row, 1, "Count",             hdr)
    ws.write(row, 2, "Mean triage score", hdr)
    ws.write(row, 3, "Mean pIC50",        hdr)
    _blank()
    if "scaffold_bucket" in df.columns:
        sc_counts = df.groupby("scaffold_bucket").size().sort_values(ascending=False).head(15)
        for scaff, cnt in sc_counts.items():
            grp = df[df["scaffold_bucket"] == scaff]
            sname = str(scaff)[:80]
            ws.write(row, 0, sname, txt_c)
            ws.write_number(row, 1, int(cnt), val_i)
            ts = pd.to_numeric(grp.get("triage_score", pd.Series(dtype=float)), errors="coerce")
            ws.write_number(row, 2, round(float(ts.mean()), 4), val)
            pi = pd.to_numeric(grp.get("pred_pIC50", pd.Series(dtype=float)), errors="coerce")
            ws.write_number(row, 3, round(float(pi.mean()), 4), val)
            _blank()
    _blank()

    # ── Section 5: ADME parameter distribution ────────────────────────────────
    ws.merge_range(row, 0, row, 5, "5. ADME PARAMETER DISTRIBUTION (PERCENTILES)", sec)
    _blank()
    ws.write(row, 0, "Parameter", hdr)
    ws.write(row, 1, "Min",       hdr)
    ws.write(row, 2, "P25",       hdr)
    ws.write(row, 3, "Median",    hdr)
    ws.write(row, 4, "P75",       hdr)
    ws.write(row, 5, "Max",       hdr)
    _blank()
    pct_cols = [
        "triage_score", "pred_pIC50", "pred_ic50_uM", "pred_cLogP",
        "pred_logS_ESOL", "mol_weight",
        "pred_MetStab_Clearance_Microsome_AZ",
        "pred_MetStab_Clearance_Hepatocyte_AZ",
        "pred_MetStab_Half_Life_Obach",
    ]
    for pc in pct_cols:
        if pc not in df.columns:
            continue
        s = pd.to_numeric(df[pc], errors="coerce").dropna()
        if s.empty:
            continue
        ws.write(row, 0, pc, lbl)
        for ci, q in enumerate([0, 0.25, 0.5, 0.75, 1.0], start=1):
            fmt = hi_v if pc in _GREEN_COLS else (lo_v if pc in _RED_COLS else val)
            ws.write_number(row, ci, round(float(s.quantile(q)), 4), fmt)
        _blank()


def write_xlsx(df: pd.DataFrame, path: Path, smiles_col: str = "canonical_smiles") -> None:
    enriched = _enrich(df, smiles_col)
    wb = xlsxwriter.Workbook(str(path), {"constant_memory": False, "nan_inf_to_errors": True})
    _add_compounds_sheet(wb, enriched, smiles_col)
    _add_analysis_sheet(wb, enriched)
    wb.close()


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = [
    "canonical_smiles",
    "pred_ic50_uM",
    "pred_pIC50",
    "pred_pIC50_uncertainty",
    "pred_cLogP",
    "pred_logS_ESOL",
    "pred_MetStab_Clearance_Microsome_AZ",
    "pred_MetStab_Clearance_Hepatocyte_AZ",
    "pred_MetStab_Half_Life_Obach",
]


def _admet_model() -> ADMETModel:
    if not hasattr(_admet_model, "_instance"):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _admet_model._instance = ADMETModel()
    return _admet_model._instance


def _safe_numeric(value: object, default: float = np.nan) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(pd.to_numeric(text.replace("<", "").replace(">", ""), errors="coerce"))
    except Exception:
        return default


def _sa_score(smi: str) -> float:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return float("nan")
    try:
        sascorer = import_module('rdkit.Contrib.SA_Score.sascorer')
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float("nan")


def _sa_accessibility(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (1.0 - ((s - 1.0) / 9.0)).clip(lower=0.0, upper=1.0).fillna(0.0)


def _fingerprint(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _embed_3d_mol(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    try:
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        return Chem.RemoveHs(mol)
    except Exception:
        return None


def _load_reference_actives(limit: int = 12) -> list[str]:
    refs: list[str] = []
    if REFERENCE_LABELED.exists():
        try:
            df = pd.read_csv(REFERENCE_LABELED)
            smiles_col = "Canonical_SMILES" if "Canonical_SMILES" in df.columns else "Canonical SMILES"
            if smiles_col in df.columns and "IC50 uM" in df.columns:
                tmp = df[[smiles_col, "IC50 uM"]].copy()
                tmp["IC50_num"] = tmp["IC50 uM"].map(lambda x: _safe_numeric(x, default=np.nan))
                tmp = tmp.dropna(subset=[smiles_col, "IC50_num"])
                tmp = tmp[tmp["IC50_num"] <= 5.0].sort_values("IC50_num", ascending=True)
                refs = [str(x) for x in tmp[smiles_col].dropna().astype(str).unique().tolist()[:limit]]
        except Exception:
            refs = []
    if not refs and REFERENCE_SMI.exists():
        refs = [line.strip().split()[0] for line in REFERENCE_SMI.read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
    return refs


def _max_tanimoto_to_refs(smiles_list: list[str], refs: list[str]) -> np.ndarray:
    ref_fps = [fp for fp in (_fingerprint(s) for s in refs) if fp is not None]
    if not ref_fps:
        return np.zeros(len(smiles_list), dtype=float)
    scores = []
    for smi in smiles_list:
        fp = _fingerprint(smi)
        if fp is None:
            scores.append(0.0)
        else:
            vals = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            scores.append(max(vals) if vals else 0.0)
    return np.asarray(scores, dtype=float)


def _max_shape_tanimoto_to_refs(smiles_list: list[str], refs: list[str]) -> np.ndarray:
    ref_mols = [mol for mol in (_embed_3d_mol(s) for s in refs) if mol is not None]
    if not ref_mols:
        return np.zeros(len(smiles_list), dtype=float)
    out = []
    for smi in smiles_list:
        mol = _embed_3d_mol(smi)
        if mol is None:
            out.append(0.0)
            continue
        best = 0.0
        for ref in ref_mols:
            try:
                sim = 1.0 - float(rdShapeHelpers.ShapeTanimotoDist(mol, ref))
            except Exception:
                sim = 0.0
            if sim > best:
                best = sim
        out.append(best)
    return np.asarray(out, dtype=float)


def _ensure_qsar_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"pred_pIC50_uncertainty", "pred_ic50_uM_upper95", "pred_pIC50_lower95"}
    if needed.issubset(df.columns):
        return df
    if not QSAR_MODEL_PATH.exists():
        return df
    bundle = joblib.load(QSAR_MODEL_PATH)
    summary = summarize_predictions(df["canonical_smiles"].astype(str).tolist(), bundle, max_models=5)
    out = df.copy()
    for key, values in summary.items():
        if key == "qsar_model_names":
            out["qsar_models_used"] = " | ".join(values)
        elif np.isscalar(values):
            out[key] = values
        else:
            out[key] = np.asarray(values)
    if "qsar_model_count" in summary:
        out["qsar_model_count"] = int(summary["qsar_model_count"])
    return out


def _ensure_target_free_admet(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TARGET_FREE_ADMET_COLUMNS if c not in df.columns]
    if not missing:
        return df
    out = df.copy()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        admet_df = _admet_model().predict(smiles=out["canonical_smiles"].astype(str).tolist()).reset_index(drop=True)
    for col in missing:
        out[col] = admet_df[col] if col in admet_df.columns else np.nan
    return out


def build_target_free_gate2(shortlist: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_target_free_admet(_ensure_qsar_uncertainty(shortlist.copy()))
    out["sa_score"] = out.get("sa_score", out["canonical_smiles"].map(_sa_score))
    out["sa_accessibility"] = _sa_accessibility(out["sa_score"])
    refs = _load_reference_actives(limit=12)
    out["nearest_active_tanimoto"] = _max_tanimoto_to_refs(out["canonical_smiles"].astype(str).tolist(), refs)
    out["shape_tanimoto_max"] = _max_shape_tanimoto_to_refs(out["canonical_smiles"].astype(str).tolist(), refs)
    alert_cols = [c for c in ["PAINS_alert", "BRENK_alert", "NIH_alert"] if c in out.columns]
    out["liability_alert_count"] = out[alert_cols].fillna(0).sum(axis=1) if alert_cols else 0.0
    out["liability_free"] = (pd.to_numeric(out["liability_alert_count"], errors="coerce").fillna(0) <= 0).astype(int)
    out["uncertainty_confidence_score"] = _normalize(out.get("pred_pIC50_uncertainty", pd.Series(np.zeros(len(out)))), higher_is_better=False)
    out["ligand_support_score"] = (
        0.65 * pd.to_numeric(out["nearest_active_tanimoto"], errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(out["shape_tanimoto_max"], errors="coerce").fillna(0.0)
    )
    out["liability_score"] = (1.0 - (pd.to_numeric(out["liability_alert_count"], errors="coerce").fillna(0.0) / 3.0)).clip(lower=0.0, upper=1.0)
    out["passes_qsar_uncertainty"] = (pd.to_numeric(out.get("pred_ic50_uM_upper95", np.nan), errors="coerce") <= 5.0).astype(int)
    out["gate2_target_free_score"] = (
        0.40 * pd.to_numeric(out.get("triage_score", 0.0), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out["uncertainty_confidence_score"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out["sa_accessibility"], errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out["liability_score"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out["ligand_support_score"], errors="coerce").fillna(0.0)
    )
    out["passes_gate2_target_free"] = (
        (pd.to_numeric(out["pred_pIC50_uncertainty"], errors="coerce").fillna(99.0) <= 0.50)
        & (pd.to_numeric(out["liability_alert_count"], errors="coerce").fillna(99.0) <= 1.0)
        & (pd.to_numeric(out["sa_score"], errors="coerce").fillna(99.0) <= 6.5)
    ).astype(int)
    out = out.sort_values(
        ["passes_gate2_target_free", "gate2_target_free_score", "pred_pIC50_uncertainty", "rank"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    out["gate2_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def build_target_free_gate3(refined: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_target_free_admet(refined.copy())
    tox_cols = [c for c in ["hERG", "DILI", "ClinTox", "AMES", "Carcinogens_Lagunin"] if c in out.columns]
    out["tox_risk_mean"] = out[tox_cols].apply(pd.to_numeric, errors="coerce").fillna(0.5).mean(axis=1) if tox_cols else 0.5
    out["tox_safety_score"] = (1.0 - pd.to_numeric(out["tox_risk_mean"], errors="coerce").fillna(0.5)).clip(lower=0.0, upper=1.0)
    cyp_cols = [c for c in ["CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith"] if c in out.columns]
    out["max_cyp_inhibition"] = out[cyp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.5).max(axis=1) if cyp_cols else 0.5
    out["ddi_safety_score"] = (1.0 - pd.to_numeric(out["max_cyp_inhibition"], errors="coerce").fillna(0.5)).clip(lower=0.0, upper=1.0)
    absorption_terms = []
    if "HIA_Hou" in out.columns:
        absorption_terms.append(_normalize(out["HIA_Hou"], higher_is_better=True))
    if "Bioavailability_Ma" in out.columns:
        absorption_terms.append(_normalize(out["Bioavailability_Ma"], higher_is_better=True))
    if "PAMPA_NCATS" in out.columns:
        absorption_terms.append(_normalize(out["PAMPA_NCATS"], higher_is_better=True))
    if "Caco2_Wang" in out.columns:
        absorption_terms.append(_normalize(out["Caco2_Wang"], higher_is_better=True))
    out["absorption_score"] = sum(absorption_terms) / len(absorption_terms) if absorption_terms else 0.5
    out["gate3_translational_score"] = (
        0.45 * pd.to_numeric(out.get("gate2_target_free_score", 0.0), errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["tox_safety_score"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out["ddi_safety_score"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out["absorption_score"], errors="coerce").fillna(0.0)
    )
    tox_cut = max(0.45, float(pd.to_numeric(out["tox_safety_score"], errors="coerce").median()))
    ddi_cut = max(0.10, float(pd.to_numeric(out["ddi_safety_score"], errors="coerce").median()))
    abs_cut = max(0.70, float(pd.to_numeric(out["absorption_score"], errors="coerce").median()))
    out["passes_gate3_translational"] = (
        (pd.to_numeric(out["tox_safety_score"], errors="coerce").fillna(0.0) >= tox_cut)
        & (pd.to_numeric(out["ddi_safety_score"], errors="coerce").fillna(0.0) >= ddi_cut)
        & (pd.to_numeric(out["absorption_score"], errors="coerce").fillna(0.0) >= abs_cut)
    ).astype(int)
    out["experimental_priority_score"] = (
        0.70 * pd.to_numeric(out["gate3_translational_score"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(out.get("gate2_target_free_score", 0.0), errors="coerce").fillna(0.0)
    )
    out = out.sort_values(
        ["passes_gate3_translational", "experimental_priority_score", "gate2_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    out["gate3_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default).astype(float)


def _normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = _to_numeric(series)
    lo = float(s.min())
    hi = float(s.max())
    if np.isclose(lo, hi):
        base = pd.Series(np.full(len(s), 0.5), index=s.index, dtype=float)
    else:
        base = (s - lo) / (hi - lo)
    return base if higher_is_better else (1.0 - base)


def _clogp_window_score(series: pd.Series, low: float = 1.0, high: float = 3.5) -> pd.Series:
    s = _to_numeric(series)
    center = (low + high) / 2.0
    half_width = (high - low) / 2.0
    d = (s - center).abs()
    score = 1.0 - (d / max(half_width, 1e-6))
    return score.clip(lower=0.0, upper=1.0)


def _sanitize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _murcko(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if scaff is None:
        return ""
    return Chem.MolToSmiles(scaff) if scaff.GetNumAtoms() > 0 else ""


def discover_scored_csvs(input_root: Path) -> list[Path]:
    paths = sorted(input_root.rglob("*_generated_scored.csv"))
    return [p for p in paths if p.is_file() and "animations" not in str(p)]


def _source_metadata(path: Path, input_root: Path) -> dict[str, str]:
    rel = path.relative_to(input_root)
    parts = rel.parts
    family = parts[0] if len(parts) >= 1 else "unknown"
    mode = parts[1] if len(parts) >= 2 else "unknown"
    return {
        "source_file": str(rel),
        "source_family": family,
        "source_mode": mode,
    }


def load_master_table(scored_csvs: Iterable[Path], input_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in scored_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "canonical_smiles" not in df.columns:
            continue
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        meta = _source_metadata(csv_path, input_root)
        for key, val in meta.items():
            df[key] = val
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source_file", "source_family", "source_mode"])

    master = pd.concat(frames, ignore_index=True)
    master["canonical_smiles"] = master["canonical_smiles"].astype(str).map(_sanitize_smiles)
    master = master.dropna(subset=["canonical_smiles"]).copy()
    master = master.drop_duplicates(subset=["canonical_smiles", "source_family", "source_mode"]).reset_index(drop=True)
    return master


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    for col in REQUIRED_COLUMNS:
        if col != "canonical_smiles":
            scored[col] = _to_numeric(scored[col])

    potency = _normalize(scored["pred_pIC50"], higher_is_better=True)
    log_s = _normalize(scored["pred_logS_ESOL"], higher_is_better=True)
    clogp = _clogp_window_score(scored["pred_cLogP"], low=1.0, high=3.5)
    micro_clear = _normalize(scored["pred_MetStab_Clearance_Microsome_AZ"], higher_is_better=False)
    hep_clear = _normalize(scored["pred_MetStab_Clearance_Hepatocyte_AZ"], higher_is_better=False)
    half_life = _normalize(scored["pred_MetStab_Half_Life_Obach"], higher_is_better=True)

    scored["triage_score"] = (
        0.35 * potency
        + 0.15 * log_s
        + 0.15 * clogp
        + 0.10 * micro_clear
        + 0.10 * hep_clear
        + 0.15 * half_life
    )

    scored["murcko_scaffold"] = scored["canonical_smiles"].map(_murcko)
    scored["murcko_scaffold"] = scored["murcko_scaffold"].fillna("").astype(str)
    scored["scaffold_bucket"] = np.where(scored["murcko_scaffold"] == "", "acyclic_or_none", scored["murcko_scaffold"])

    scored = scored.sort_values(["triage_score", "pred_pIC50"], ascending=[False, False]).reset_index(drop=True)
    return scored


def select_diverse_shortlist(df: pd.DataFrame, top_k: int, max_per_scaffold: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    selected_idx: list[int] = []
    scaffold_counts: dict[str, int] = {}

    for idx, row in df.iterrows():
        scaffold = str(row["scaffold_bucket"])
        count = scaffold_counts.get(scaffold, 0)
        if count >= max_per_scaffold:
            continue
        selected_idx.append(idx)
        scaffold_counts[scaffold] = count + 1
        if len(selected_idx) >= top_k:
            break

    shortlist = df.loc[selected_idx].copy().reset_index(drop=True)
    shortlist.insert(0, "rank", np.arange(1, len(shortlist) + 1, dtype=int))
    return shortlist


def write_gate1(scored: pd.DataFrame, shortlist: pd.DataFrame, out_root: Path) -> dict:
    gate1 = out_root / "gate1"
    gate1.mkdir(parents=True, exist_ok=True)

    master_path = gate1 / "gate1_master_ranked.csv"
    shortlist_path = gate1 / "gate1_shortlist.csv"
    summary_path = gate1 / "gate1_summary.json"

    scored.to_csv(master_path, index=False)
    shortlist.to_csv(shortlist_path, index=False)

    shortlist_xlsx = gate1 / "gate1_shortlist.xlsx"
    write_xlsx(shortlist, shortlist_xlsx)

    summary = {
        "master_candidates": int(len(scored)),
        "shortlist_candidates": int(len(shortlist)),
        "unique_scaffolds_master": int(scored["scaffold_bucket"].nunique()) if not scored.empty else 0,
        "unique_scaffolds_shortlist": int(shortlist["scaffold_bucket"].nunique()) if not shortlist.empty else 0,
        "source_families": sorted(scored["source_family"].dropna().astype(str).unique().tolist()) if not scored.empty else [],
        "source_modes": sorted(scored["source_mode"].dropna().astype(str).unique().tolist()) if not scored.empty else [],
        "artifacts": {
            "master_ranked_csv": str(master_path.relative_to(ROOT)),
            "shortlist_csv": str(shortlist_path.relative_to(ROOT)),
            "shortlist_xlsx": str(shortlist_xlsx.relative_to(ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_gate2(shortlist: pd.DataFrame, out_root: Path) -> None:
    gate2 = out_root / "gate2_target_free_refinement"
    gate2.mkdir(parents=True, exist_ok=True)

    csv_path = gate2 / "target_free_candidates.csv"
    readme = gate2 / "README_GATE2.txt"

    keep_cols = [
        "gate2_rank",
        "rank",
        "canonical_smiles",
        "triage_score",
        "gate2_target_free_score",
        "passes_gate2_target_free",
        "pred_pIC50_uncertainty",
        "pred_pIC50_lower95",
        "pred_ic50_uM_upper95",
        "uncertainty_confidence_score",
        "sa_score",
        "sa_accessibility",
        "liability_alert_count",
        "liability_free",
        "nearest_active_tanimoto",
        "shape_tanimoto_max",
        "ligand_support_score",
        "pred_pIC50",
        "pred_ic50_uM",
        "pred_cLogP",
        "pred_logS_ESOL",
        "pred_MetStab_Clearance_Microsome_AZ",
        "pred_MetStab_Clearance_Hepatocyte_AZ",
        "pred_MetStab_Half_Life_Obach",
        "scaffold_bucket",
        "source_family",
        "source_mode",
    ]
    out_df = shortlist[[c for c in keep_cols if c in shortlist.columns]].copy()
    out_df.to_csv(csv_path, index=False)
    write_xlsx(out_df, gate2 / "target_free_candidates.xlsx")

    readme.write_text(
        "Gate 2 target-free refinement\n"
        "\n"
        "Inputs generated by run_hybrid_stage_gates.py\n"
        "- target_free_candidates.csv: reranked shortlist using uncertainty-aware ligand-based filters\n"
        "- target_free_candidates.xlsx: same data with structures and analysis\n"
        "\n"
        "Implemented filters:\n"
        "1) QSAR uncertainty from up to five positive-R² models\n"
        "2) Synthetic accessibility (SA score)\n"
        "3) PAINS/BRENK/NIH alert filtering\n"
        "4) Ligand-based 2D Tanimoto to known actives\n"
        "5) Ligand-based 3D shape similarity to known actives\n"
        "\n"
        "Suggested use:\n"
        "Rank by gate2_target_free_score and prioritize passes_gate2_target_free=1\n",
        encoding="utf-8",
    )


def write_gate3(shortlist: pd.DataFrame, out_root: Path) -> None:
    gate3 = out_root / "gate3_target_free_translational"
    gate3.mkdir(parents=True, exist_ok=True)

    cand_path = gate3 / "translational_candidates.csv"
    template_path = gate3 / "experimental_shortlist_template.csv"
    readme = gate3 / "README_GATE3.txt"

    keep_cols = [
        "gate3_rank",
        "gate2_rank",
        "rank",
        "canonical_smiles",
        "experimental_priority_score",
        "gate3_translational_score",
        "passes_gate3_translational",
        "gate2_target_free_score",
        "tox_safety_score",
        "ddi_safety_score",
        "absorption_score",
        "hERG",
        "DILI",
        "ClinTox",
        "AMES",
        "Carcinogens_Lagunin",
        "max_cyp_inhibition",
        "HIA_Hou",
        "Bioavailability_Ma",
        "PAMPA_NCATS",
        "Caco2_Wang",
        "pred_pIC50",
        "pred_ic50_uM",
        "pred_pIC50_uncertainty",
        "source_family",
        "source_mode",
    ]
    out_df = shortlist[[c for c in keep_cols if c in shortlist.columns]].copy()
    out_df.to_csv(cand_path, index=False)
    write_xlsx(out_df, gate3 / "translational_candidates.xlsx")

    template = pd.DataFrame([
        {
            "parameter": "batch_priority",
            "value": "top_25_first",
            "notes": "start wet-lab triage from highest experimental_priority_score",
        },
        {
            "parameter": "uncertainty_rule",
            "value": "prefer low pred_pIC50_uncertainty",
            "notes": "avoid highly uncertain QSAR extrapolations",
        },
        {
            "parameter": "chemistry_rule",
            "value": "exclude alerting chemotypes",
            "notes": "do not prioritize PAINS/BRENK/NIH-positive molecules",
        },
        {
            "parameter": "assay_panel",
            "value": "potency+microsome+hepato+hERG",
            "notes": "recommended minimal validation panel",
        },
        {
            "parameter": "progression_rule",
            "value": "requires passes_gate3_translational=1",
            "notes": "default gate for experimental shortlist",
        },
    ])
    template.to_csv(template_path, index=False)

    readme.write_text(
        "Gate 3 target-free translational prioritization\n"
        "\n"
        "Generated files:\n"
        "- translational_candidates.csv: final reranked candidates for experimental planning\n"
        "- translational_candidates.xlsx: same data with structures and analysis\n"
        "- experimental_shortlist_template.csv: wet-lab planning template\n"
        "\n"
        "Implemented target-free translational signals:\n"
        "1) Tox liabilities: hERG, DILI, ClinTox, AMES, carcinogenicity\n"
        "2) DDI liabilities: CYP1A2/2C19/2C9/2D6/3A4 inhibition risk\n"
        "3) Absorption support: HIA, bioavailability, PAMPA, Caco2\n"
        "4) Final experimental priority score for lab selection\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run stage-gated hybrid pipeline artifacts from scored outputs.")
    p.add_argument("--input-root", default=str(ROOT / "outputs" / "generated"))
    p.add_argument("--output-root", default=str(ROOT / "outputs" / "hybrid_stage"))
    p.add_argument("--top-k", type=int, default=200,
        help="Number of candidates in Gate 1 shortlist (default: 200).")
    p.add_argument("--max-per-scaffold", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    scored_csvs = discover_scored_csvs(input_root)
    if not scored_csvs:
        print(f"No scored CSV files found under: {input_root}")
        return 1

    master = load_master_table(scored_csvs, input_root)
    if master.empty:
        print("No valid candidates found in discovered scored files.")
        return 1

    ranked = score_candidates(master)
    ranked = ranked.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    shortlist = select_diverse_shortlist(ranked, top_k=max(1, args.top_k), max_per_scaffold=max(1, args.max_per_scaffold))

    summary = write_gate1(ranked, shortlist, output_root)
    gate2_shortlist = build_target_free_gate2(shortlist)
    gate3_shortlist = build_target_free_gate3(gate2_shortlist)
    write_gate2(gate2_shortlist, output_root)
    write_gate3(gate3_shortlist, output_root)

    run_manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "discovered_scored_files": len(scored_csvs),
        "gate1_summary": summary,
        "gate2_dir": str((output_root / "gate2_target_free_refinement").relative_to(ROOT)),
        "gate3_dir": str((output_root / "gate3_target_free_translational").relative_to(ROOT)),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print("Hybrid stage-gated artifacts created:")
    print(f"- {output_root / 'gate1' / 'gate1_master_ranked.csv'}")
    print(f"- {output_root / 'gate1' / 'gate1_shortlist.csv'}")
    print(f"- {output_root / 'gate2_target_free_refinement'}")
    print(f"- {output_root / 'gate3_target_free_translational'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
