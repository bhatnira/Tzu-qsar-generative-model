#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from animation_metrics import load_table, metric_lines, safe_render_molecule_image

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PLAIN_PATH = ROOT / "outputs/generated/mol2mol/plain/mol2mol_plain_generated_scored.xlsx"
RL_PATH = ROOT / "outputs/generated/mol2mol/reinforcement_learning/mol2mol_rl_generated_scored.xlsx"
CL_PATH = ROOT / "outputs/generated/mol2mol/curriculum_learning/mol2mol_cl_generated_scored.xlsx"
OUT_PATH = ROOT / "outputs/generated/animations/final_structures/minimal_diffs/preserved_scaffold_all_in_one.gif"


def parse_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip() or smiles.lower() == "nan":
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None



def safe_scaffold(smiles: str) -> Optional[str]:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return None



def load_df(path: Path) -> pd.DataFrame:
    df = load_table(path)
    if "SMILES" not in df.columns and "canonical_smiles" in df.columns:
        df = df.rename(columns={"canonical_smiles": "SMILES"})
    return df[df["Input_SMILES"].astype(str).ne("nan") & df["SMILES"].astype(str).ne("nan")].copy()



def first_row_map(df: pd.DataFrame) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        inp = str(row["Input_SMILES"])
        smi = str(row["SMILES"])
        if inp not in out and parse_smiles(smi) is not None:
            out[inp] = row.to_dict()
    return out



def preserved_inputs(plain_df: pd.DataFrame, rl_df: pd.DataFrame, cl_df: pd.DataFrame) -> List[str]:
    plain_map = first_row_map(plain_df)
    rl_map = first_row_map(rl_df)
    keep: List[str] = []
    seen = set()

    for _, row in cl_df.iterrows():
        inp = str(row["Input_SMILES"])
        out = str(row["SMILES"])
        if inp in seen or inp not in plain_map or inp not in rl_map:
            continue
        if parse_smiles(out) is None:
            continue
        in_scaf = safe_scaffold(inp)
        out_scaf = safe_scaffold(out)
        if in_scaf and out_scaf and in_scaf == out_scaf:
            keep.append(inp)
            seen.add(inp)
    return keep



def render_panel(smiles: str, title: str, row: Optional[dict] = None, panel_size=(330, 430)) -> Image.Image:
    panel = Image.new("RGB", panel_size, "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    if parse_smiles(smiles) is not None:
        img = safe_render_molecule_image(smiles, 300, 250)
        panel.paste(img, (15, 45))
    else:
        draw.text((15, 160), "Invalid SMILES", fill="black", font=font)

    draw.text((10, 10), title, fill="black", font=font)
    for i, line in enumerate(metric_lines(smiles, row)):
        draw.text((10, 342 + i * 16), line, fill="black", font=font)
    draw.rectangle([0, 0, panel_size[0] - 1, panel_size[1] - 1], outline=(175, 175, 175))
    return panel



def build_frame(idx: int, total: int, inp: str, plain_row: dict, rl_row: dict, cl_row: dict) -> Image.Image:
    W, H = 1460, 620
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    scaffold = safe_scaffold(inp) or "n/a"
    draw.text((20, 15), "All Preserved Scaffold Cases: Input -> Plain -> RL -> CL", fill="black", font=font)
    draw.text((20, 36), f"Case {idx + 1}/{total}", fill=(80, 80, 80), font=font)
    draw.text((20, 56), f"Scaffold: {scaffold[:120]}", fill=(40, 40, 40), font=font)

    stages = [
        ("Input", inp, {"SMILES": inp}),
        ("Plain", str(plain_row["SMILES"]), plain_row),
        ("RL", str(rl_row["SMILES"]), rl_row),
        ("CL", str(cl_row["SMILES"]), cl_row),
    ]

    for i, (label, smi, row) in enumerate(stages):
        x = 20 + i * 355
        y = 95
        panel = render_panel(smi, label, row=row)
        canvas.paste(panel, (x, y))

    return canvas



def main() -> None:
    plain_df = load_df(PLAIN_PATH)
    rl_df = load_df(RL_PATH)
    cl_df = load_df(CL_PATH)

    plain_map = first_row_map(plain_df)
    rl_map = first_row_map(rl_df)
    cl_map = first_row_map(cl_df)
    inputs = preserved_inputs(plain_df, rl_df, cl_df)

    if not inputs:
        raise RuntimeError("No preserved scaffold inputs found")

    frames: List[Image.Image] = []
    total = len(inputs)
    for idx, inp in enumerate(inputs):
        frames.append(build_frame(idx, total, inp, plain_map[inp], rl_map[inp], cl_map[inp]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=[900] * (len(frames) - 1) + [1500],
        loop=0,
        optimize=True,
    )
    print(f"Created: {OUT_PATH}")
    print(f"Cases: {total}")


if __name__ == "__main__":
    main()
