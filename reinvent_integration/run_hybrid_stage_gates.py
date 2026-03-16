#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from importlib import import_module
from pathlib import Path
from shutil import which
import subprocess
import sys
import tempfile
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from pandas.errors import ParserError
import xlsxwriter
from admet_ai import ADMETModel
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdShapeHelpers
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Crippen, Descriptors
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
    "molecular_weight", "logP",
]

_GREEN_COLS = {"passes_optimal", "pred_pIC50", "pred_MetStab_Half_Life_Obach", "triage_score", "experimental_priority_score", "uncertainty_confidence_score", "liability_free"}
_RED_COLS   = {"pred_ic50_uM", "pred_MetStab_Clearance_Microsome_AZ", "pred_MetStab_Clearance_Hepatocyte_AZ", "pred_pIC50_uncertainty", "hERG", "DILI", "ClinTox", "AMES", "max_cyp_inhibition"}

_GREEN_COLS.update({
    "clogp_score_individual",
    "clogp_score_qikprop",
    "clogp_ensemble_score",
    "solubility_score_individual",
    "solubility_score_qikprop",
    "solubility_ensemble_score",
    "microsome_stability_score",
    "hepatocyte_stability_score",
    "half_life_stability_score",
    "metabolic_stability_ensemble_score",
    "adme_individual_score",
    "adme_ensemble_score",
    "passes_adme_individual",
    "passes_adme_ensemble",
    "passes_metabolic_stability",
    "osp_translational_score",
})
_RED_COLS.update({
    "qikprop_star_count",
    "qikprop_metab_sites",
    "clogp_uncertainty",
    "solubility_uncertainty",
    "metstab_uncertainty",
    "adme_ensemble_uncertainty",
})

_ADME_CUTS = {
    "pred_ic50_uM":                           ("<=", 5.0),
    "pred_cLogP":                             ("1-3.5", None),
    "pred_logS_ESOL":                         (">=", -6.0),
    "pred_MetStab_Clearance_Microsome_AZ":    ("<=", 70.0),
    "pred_MetStab_Clearance_Hepatocyte_AZ":   ("<=", 80.0),
    "pred_MetStab_Half_Life_Obach":           (">=", 40.0),
    "clogp_ensemble_value":                   ("1-3.5", None),
    "solubility_ensemble_value":              (">=", -6.0),
}

QIKPROP_EXECUTABLE_CANDIDATES = [
    "/opt/schrodinger/schrodinger2026-1/qikprop",
    "qikprop",
]

QIKPROP_COLUMN_RENAME = {
    "#stars": "qikprop_star_count",
    "QPlogPo/w": "qikprop_clogp",
    "QPlogS": "qikprop_logS",
    "#metab": "qikprop_metab_sites",
    "HumanOralAbsorption": "qikprop_human_oral_absorption_class",
    "PercentHumanOralAbsorption": "qikprop_percent_human_oral_absorption",
    "QPPCaco": "qikprop_caco2",
}

QIKPROP_COLUMN_ALIASES = {
    "qikprop_star_count": ["#stars", "qikprop_star_count"],
    "qikprop_clogp": ["QPlogPo/w", "qikprop_clogp"],
    "qikprop_logS": ["QPlogS", "qikprop_logs"],
    "qikprop_metab_sites": ["#metab", "qikprop_metab_sites"],
    "qikprop_human_oral_absorption_class": ["HumanOralAbsorption", "qikprop_human_oral_absorption_class"],
    "qikprop_percent_human_oral_absorption": ["PercentHumanOralAbsorption", "qikprop_percent_human_oral_absorption"],
    "qikprop_caco2": ["QPPCaco", "qikprop_caco2"],
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


def _series_for(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.full(len(df), default), index=df.index, dtype=float)


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


def _high_better_score(series: pd.Series, threshold: float, scale: float) -> pd.Series:
    s = _to_numeric(series)
    score = 0.5 + ((s - threshold) / max(scale, 1e-6)) * 0.5
    return score.clip(lower=0.0, upper=1.0)


def _low_better_score(series: pd.Series, threshold: float, scale: float) -> pd.Series:
    s = _to_numeric(series)
    score = 0.5 + ((threshold - s) / max(scale, 1e-6)) * 0.5
    return score.clip(lower=0.0, upper=1.0)


def _mean_available(columns: list[pd.Series], index: pd.Index) -> pd.Series:
    if not columns:
        return pd.Series(np.full(len(index), np.nan), index=index, dtype=float)
    frame = pd.concat(columns, axis=1)
    return frame.apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def _std_available(columns: list[pd.Series], index: pd.Index) -> pd.Series:
    if not columns:
        return pd.Series(np.full(len(index), 0.0), index=index, dtype=float)
    frame = pd.concat(columns, axis=1).apply(pd.to_numeric, errors="coerce")
    std = frame.std(axis=1, skipna=True, ddof=0).fillna(0.0)
    return std.clip(lower=0.0, upper=1.0)


def _rdkit_proxy_series(smiles: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    clogp, logs_proxy, metstab_proxy = [], [], []
    for smi in smiles.astype(str).tolist():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            clogp.append(np.nan)
            logs_proxy.append(np.nan)
            metstab_proxy.append(np.nan)
            continue
        mw = float(Descriptors.MolWt(mol))
        lp = float(Crippen.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        rot = float(Descriptors.NumRotatableBonds(mol))
        logs_est = -0.012 * mw - 0.55 * lp + 0.003 * tpsa - 1.20
        met_est = 70.0 - 7.5 * lp - 1.2 * rot + 0.12 * tpsa
        clogp.append(lp)
        logs_proxy.append(logs_est)
        metstab_proxy.append(max(1.0, min(120.0, met_est)))
    idx = smiles.index
    return (
        pd.Series(clogp, index=idx, dtype=float),
        pd.Series(logs_proxy, index=idx, dtype=float),
        pd.Series(metstab_proxy, index=idx, dtype=float),
    )


def _resolve_qikprop_executable() -> str | None:
    if hasattr(_resolve_qikprop_executable, "_cached"):
        return _resolve_qikprop_executable._cached
    resolved = None
    for candidate in QIKPROP_EXECUTABLE_CANDIDATES:
        if "/" in candidate:
            path = Path(candidate)
            if path.exists() and path.is_file():
                resolved = str(path)
                break
        else:
            hit = which(candidate)
            if hit:
                resolved = hit
                break
    _resolve_qikprop_executable._cached = resolved
    return resolved


def _norm_qikprop_col(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _resolve_qikprop_columns(columns: list[str]) -> dict[str, str]:
    norm_to_col = {_norm_qikprop_col(col): col for col in columns}
    resolved: dict[str, str] = {}
    for target_col, aliases in QIKPROP_COLUMN_ALIASES.items():
        for alias in aliases:
            hit = norm_to_col.get(_norm_qikprop_col(alias))
            if hit is not None:
                resolved[target_col] = hit
                break
    return resolved


def _qikprop_row_index(row: pd.Series) -> int | None:
    raw = row.get("molecule", row.get("title", row.get("_Name", "")))
    if pd.isna(raw):
        return None
    match = re.search(r"(\d+)$", str(raw))
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
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _run_qikprop_predictions(smiles_list: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(smiles_list), dtype=int))
    for col in QIKPROP_COLUMN_RENAME.values():
        out[col] = np.nan
    out["qikprop_status"] = "not_run"

    executable = _resolve_qikprop_executable()
    if not executable or not smiles_list:
        out["qikprop_status"] = "unavailable"
        return out

    try:
        with tempfile.TemporaryDirectory(prefix="qikprop_gate_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            sdf_path = tmpdir / "input.sdf"
            kept_index: list[int] = []
            writer = Chem.SDWriter(str(sdf_path))
            for idx, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    continue
                mol.SetProp("_Name", f"Molecule_{idx + 1:05d}")
                writer.write(mol)
                kept_index.append(idx)
            writer.close()

            if not kept_index:
                out["qikprop_status"] = "no_valid_molecules"
                return out

            run = subprocess.run(
                [executable, "-WAIT", "-nosim", sdf_path.name],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            csv_path = tmpdir / "input.CSV"
            if not csv_path.exists() and run.returncode != 0:
                out["qikprop_status"] = f"failed:{run.returncode}"
                return out
            if not csv_path.exists():
                out["qikprop_status"] = "missing_csv"
                return out

            qdf = _read_qikprop_csv(csv_path)
            resolved_cols = _resolve_qikprop_columns(list(qdf.columns))
            if not resolved_cols:
                out["qikprop_status"] = "missing_columns"
                return out

            assigned = [False] * len(smiles_list)
            processed = [False] * len(smiles_list)
            for pos in range(len(qdf)):
                row = qdf.iloc[pos]
                row_idx = _qikprop_row_index(row)
                if row_idx is None or row_idx < 0 or row_idx >= len(smiles_list):
                    if pos < len(kept_index):
                        row_idx = kept_index[pos]
                    else:
                        continue
                for target_col, source_col in resolved_cols.items():
                    out.at[row_idx, target_col] = row.get(source_col, np.nan)
                c_val = pd.to_numeric(out.at[row_idx, "qikprop_clogp"], errors="coerce") if "qikprop_clogp" in out.columns else np.nan
                s_val = pd.to_numeric(out.at[row_idx, "qikprop_logS"], errors="coerce") if "qikprop_logS" in out.columns else np.nan
                m_val = pd.to_numeric(out.at[row_idx, "qikprop_metab_sites"], errors="coerce") if "qikprop_metab_sites" in out.columns else np.nan
                has_numeric = np.isfinite(c_val) or np.isfinite(s_val) or np.isfinite(m_val)
                out.at[row_idx, "qikprop_status"] = "ok" if has_numeric else "no_prediction"
                assigned[row_idx] = has_numeric
                processed[row_idx] = True

            for row_idx in kept_index:
                if not processed[row_idx]:
                    out.at[row_idx, "qikprop_status"] = "partial"
            return out
    except Exception as exc:
        out["qikprop_status"] = f"error:{type(exc).__name__}"
        return out


def _attach_focus_adme_scores(df: pd.DataFrame, include_qikprop: bool) -> pd.DataFrame:
    out = df.copy()

    pred_clogp = _series_for(out, "pred_cLogP")
    pred_logs = _series_for(out, "pred_logS_ESOL")
    micro = _series_for(out, "pred_MetStab_Clearance_Microsome_AZ")
    hep = _series_for(out, "pred_MetStab_Clearance_Hepatocyte_AZ")
    half_life = _series_for(out, "pred_MetStab_Half_Life_Obach")
    admet_logp = _series_for(out, "logP")
    admet_mw = _series_for(out, "molecular_weight")
    rdkit_clogp, rdkit_logs_proxy, rdkit_met_proxy = _rdkit_proxy_series(out["canonical_smiles"])

    out["clogp_score_individual"] = _clogp_window_score(pred_clogp, low=1.0, high=3.5)
    out["solubility_score_individual"] = _high_better_score(pred_logs, threshold=-6.0, scale=2.0)
    out["microsome_stability_score"] = _low_better_score(micro, threshold=70.0, scale=70.0)
    out["hepatocyte_stability_score"] = _low_better_score(hep, threshold=80.0, scale=80.0)
    out["half_life_stability_score"] = _high_better_score(half_life, threshold=40.0, scale=40.0)
    out["metabolic_stability_ensemble_score"] = (
        0.35 * pd.to_numeric(out["microsome_stability_score"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(out["hepatocyte_stability_score"], errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(out["half_life_stability_score"], errors="coerce").fillna(0.0)
    )

    if include_qikprop:
        drop_cols = [c for c in list(QIKPROP_COLUMN_RENAME.values()) + ["qikprop_status"] if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        qikprop = _run_qikprop_predictions(out["canonical_smiles"].astype(str).tolist())
        out = pd.concat([out.reset_index(drop=True), qikprop.reset_index(drop=True)], axis=1)
    else:
        for col in QIKPROP_COLUMN_RENAME.values():
            if col not in out.columns:
                out[col] = np.nan
        if "qikprop_status" not in out.columns:
            out["qikprop_status"] = "not_run"

    qikprop_clogp = _series_for(out, "qikprop_clogp")
    qikprop_logs = _series_for(out, "qikprop_logS")
    qikprop_metab = _series_for(out, "qikprop_metab_sites")
    admet_sol_proxy = -0.012 * admet_mw - 0.55 * admet_logp - 1.20
    adme_py_met_proxy = 70.0 - 8.0 * pred_clogp + 2.0 * pred_logs
    out["clogp_score_qikprop"] = _clogp_window_score(qikprop_clogp, low=1.0, high=3.5)
    out["solubility_score_qikprop"] = _high_better_score(qikprop_logs, threshold=-6.0, scale=2.0)
    clogp_score_admet = _clogp_window_score(admet_logp, low=1.0, high=3.5)
    clogp_score_rdkit = _clogp_window_score(rdkit_clogp, low=1.0, high=3.5)
    sol_score_admet_proxy = _high_better_score(admet_sol_proxy, threshold=-6.0, scale=2.0)
    sol_score_rdkit = _high_better_score(rdkit_logs_proxy, threshold=-6.0, scale=2.0)
    met_score_qikprop = _low_better_score(qikprop_metab, threshold=3.0, scale=2.0)
    met_score_rdkit = _high_better_score(rdkit_met_proxy, threshold=40.0, scale=20.0)
    met_score_adme_py_proxy = _high_better_score(adme_py_met_proxy, threshold=40.0, scale=20.0)

    out["clogp_ensemble_value"] = _mean_available([pred_clogp, qikprop_clogp, rdkit_clogp, admet_logp], out.index)
    out["solubility_ensemble_value"] = _mean_available([pred_logs, qikprop_logs, rdkit_logs_proxy, admet_sol_proxy], out.index)
    out["clogp_ensemble_score"] = _mean_available([
        pd.to_numeric(out["clogp_score_individual"], errors="coerce"),
        pd.to_numeric(out["clogp_score_qikprop"], errors="coerce"),
        pd.to_numeric(clogp_score_admet, errors="coerce"),
        pd.to_numeric(clogp_score_rdkit, errors="coerce"),
    ], out.index).fillna(pd.to_numeric(out["clogp_score_individual"], errors="coerce")).clip(lower=0.0, upper=1.0)
    out["solubility_ensemble_score"] = _mean_available([
        pd.to_numeric(out["solubility_score_individual"], errors="coerce"),
        pd.to_numeric(out["solubility_score_qikprop"], errors="coerce"),
        pd.to_numeric(sol_score_admet_proxy, errors="coerce"),
        pd.to_numeric(sol_score_rdkit, errors="coerce"),
    ], out.index).fillna(pd.to_numeric(out["solubility_score_individual"], errors="coerce")).clip(lower=0.0, upper=1.0)
    out["metabolic_stability_ensemble_score"] = _mean_available([
        pd.to_numeric(out["metabolic_stability_ensemble_score"], errors="coerce"),
        pd.to_numeric(met_score_qikprop, errors="coerce"),
        pd.to_numeric(met_score_rdkit, errors="coerce"),
        pd.to_numeric(met_score_adme_py_proxy, errors="coerce"),
    ], out.index).fillna(pd.to_numeric(out["metabolic_stability_ensemble_score"], errors="coerce")).clip(lower=0.0, upper=1.0)

    out["clogp_uncertainty"] = _std_available([
        pd.to_numeric(out["clogp_score_individual"], errors="coerce"),
        pd.to_numeric(out["clogp_score_qikprop"], errors="coerce"),
        pd.to_numeric(clogp_score_admet, errors="coerce"),
        pd.to_numeric(clogp_score_rdkit, errors="coerce"),
    ], out.index)
    out["solubility_uncertainty"] = _std_available([
        pd.to_numeric(out["solubility_score_individual"], errors="coerce"),
        pd.to_numeric(out["solubility_score_qikprop"], errors="coerce"),
        pd.to_numeric(sol_score_admet_proxy, errors="coerce"),
        pd.to_numeric(sol_score_rdkit, errors="coerce"),
    ], out.index)
    out["metstab_uncertainty"] = _std_available([
        pd.to_numeric(out["metabolic_stability_ensemble_score"], errors="coerce"),
        pd.to_numeric(met_score_qikprop, errors="coerce"),
        pd.to_numeric(met_score_rdkit, errors="coerce"),
        pd.to_numeric(met_score_adme_py_proxy, errors="coerce"),
    ], out.index)
    out["adme_ensemble_uncertainty"] = (
        pd.to_numeric(out["clogp_uncertainty"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["solubility_uncertainty"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["metstab_uncertainty"], errors="coerce").fillna(0.0)
    ) / 3.0

    out["adme_individual_score"] = (
        0.35 * pd.to_numeric(out["clogp_score_individual"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["solubility_score_individual"], errors="coerce").fillna(0.0)
        + 0.40 * pd.to_numeric(out["metabolic_stability_ensemble_score"], errors="coerce").fillna(0.0)
    )
    out["adme_ensemble_score"] = (
        0.35 * pd.to_numeric(out["clogp_ensemble_score"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out["solubility_ensemble_score"], errors="coerce").fillna(0.0)
        + 0.40 * pd.to_numeric(out["metabolic_stability_ensemble_score"], errors="coerce").fillna(0.0)
    )

    out["passes_metabolic_stability"] = (
        (micro <= 70.0)
        & (hep <= 80.0)
        & (half_life >= 40.0)
    ).astype(int)
    out["passes_adme_individual"] = (
        pred_clogp.between(1.0, 3.5, inclusive="both")
        & (pred_logs >= -6.0)
        & (pd.to_numeric(out["passes_metabolic_stability"], errors="coerce").fillna(0).astype(int) == 1)
    ).astype(int)
    out["passes_adme_ensemble"] = (
        pd.to_numeric(out["clogp_ensemble_value"], errors="coerce").between(1.0, 3.5, inclusive="both")
        & (pd.to_numeric(out["solubility_ensemble_value"], errors="coerce") >= -6.0)
        & (pd.to_numeric(out["passes_metabolic_stability"], errors="coerce").fillna(0).astype(int) == 1)
    ).astype(int)
    return out


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

    scored = _attach_focus_adme_scores(scored, include_qikprop=False)

    potency = _normalize(scored["pred_pIC50"], higher_is_better=True)

    scored["triage_score"] = (
        0.35 * potency
        + 0.65 * pd.to_numeric(scored["adme_individual_score"], errors="coerce").fillna(0.0)
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

    run_manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "discovered_scored_files": len(scored_csvs),
        "gate1_summary": summary,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print("Hybrid shortlist artifacts created:")
    print(f"- {output_root / 'gate1' / 'gate1_master_ranked.csv'}")
    print(f"- {output_root / 'gate1' / 'gate1_shortlist.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
