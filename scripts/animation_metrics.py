#!/usr/bin/env python3
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parents[1]
SERIES_CE_PATH = ROOT / "reinvent_integration/data/series_ce_labeled.csv"

ACTUAL_IC50_CANDIDATE_COLS = [
    "Canonical_SMILES",
    "Clean_SMILES",
    "Canonical SMILES",
    "Canonical SMILES.1",
]


@lru_cache(maxsize=20000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip() or smiles.lower() == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


@lru_cache(maxsize=20000)
def molecular_weight(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


@lru_cache(maxsize=1)
def actual_ic50_map() -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not SERIES_CE_PATH.exists():
        return mapping

    df = pd.read_csv(SERIES_CE_PATH)
    if "IC50 uM" not in df.columns:
        return mapping

    ic50_vals = pd.to_numeric(df["IC50 uM"], errors="coerce")
    for col in ACTUAL_IC50_CANDIDATE_COLS:
        if col not in df.columns:
            continue
        for smiles, ic50 in zip(df[col].astype(str), ic50_vals):
            if pd.isna(ic50):
                continue
            canon = canonicalize_smiles(smiles)
            if canon and canon not in mapping:
                mapping[canon] = float(ic50)
    return mapping


def lookup_actual_ic50(smiles: str) -> Optional[float]:
    canon = canonicalize_smiles(smiles)
    if canon is None:
        return None
    return actual_ic50_map().get(canon)



def safe_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None



def format_metric(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"



def summarize_metrics(smiles: str, row: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[float] | str]:
    row = row or {}
    return {
        "pred_ic50_uM": safe_float(row.get("pred_ic50_uM")),
        "pred_pIC50": safe_float(row.get("pred_pIC50")),
        "actual_ic50_uM": lookup_actual_ic50(smiles),
        "pred_cLogP": safe_float(row.get("pred_cLogP")),
        "pred_logS_ESOL": safe_float(row.get("pred_logS_ESOL")),
        "pred_MetStab_Clearance_Microsome_AZ": safe_float(row.get("pred_MetStab_Clearance_Microsome_AZ")),
        "pred_MetStab_Clearance_Hepatocyte_AZ": safe_float(row.get("pred_MetStab_Clearance_Hepatocyte_AZ")),
        "pred_MetStab_Half_Life_Obach": safe_float(row.get("pred_MetStab_Half_Life_Obach")),
        "mw": molecular_weight(smiles),
        "pred_Solubility_ESOL_class": row.get("pred_Solubility_ESOL_class") if row.get("pred_Solubility_ESOL_class") is not None else "n/a",
    }



def metric_lines(smiles: str, row: Optional[Dict[str, Any]] = None) -> list[str]:
    m = summarize_metrics(smiles, row)
    return [
        f"Pred IC50: {format_metric(m['pred_ic50_uM'])} uM | Actual IC50: {format_metric(m['actual_ic50_uM'])} uM",
        f"Pred pIC50: {format_metric(m['pred_pIC50'])} | MW: {format_metric(m['mw'])}",
        f"cLogP: {format_metric(m['pred_cLogP'])} | logS: {format_metric(m['pred_logS_ESOL'])}",
        f"Mic: {format_metric(m['pred_MetStab_Clearance_Microsome_AZ'])} | Hep: {format_metric(m['pred_MetStab_Clearance_Hepatocyte_AZ'])}",
        f"Half-life: {format_metric(m['pred_MetStab_Half_Life_Obach'])} | Sol: {m['pred_Solubility_ESOL_class']}",
    ]



def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)



def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def safe_render_molecule_image(smiles: str, width: int, height: int) -> Image.Image:
    placeholder = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(placeholder)
    if not isinstance(smiles, str) or not smiles.strip():
        draw.text((10, 10), "No structure", fill="black")
        return placeholder

    script = """
import sys
from rdkit import Chem
from rdkit.Chem import Draw

smiles = sys.argv[1]
out_path = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise SystemExit(2)
img = Draw.MolToImage(mol, size=(width, height), kekulize=False)
img.save(out_path)
"""

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script, smiles, str(tmp_path), str(width), str(height)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            with Image.open(tmp_path) as img:
                return img.convert("RGB")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    draw.text((10, 10), "Render unavailable", fill="black")
    return placeholder
