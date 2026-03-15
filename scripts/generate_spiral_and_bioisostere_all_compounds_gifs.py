#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from animation_metrics import load_table, metric_lines, safe_render_molecule_image

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PLAIN_CSV = ROOT / "outputs/generated/mol2mol/plain/mol2mol_plain_generated_scored.xlsx"
RL_CSV = ROOT / "outputs/generated/mol2mol/reinforcement_learning/mol2mol_rl_generated_scored.xlsx"
CL_CSV = ROOT / "outputs/generated/mol2mol/curriculum_learning/mol2mol_cl_generated_scored.xlsx"
OUT_DIR = ROOT / "outputs/generated/animations/final_structures/minimal_diffs"

NITRO_PATTERN = "[N+](=O)[O-]"
PART_SIZE = 200



def parse_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip() or smiles.lower() == "nan":
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None



def murcko_scaffold_smiles(smiles: str) -> Optional[str]:
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return None



def has_nitro(smiles: str) -> bool:
    return isinstance(smiles, str) and NITRO_PATTERN in smiles



def load_df(path: Path) -> pd.DataFrame:
    df = load_table(path)
    if "SMILES" not in df.columns and "canonical_smiles" in df.columns:
        df = df.rename(columns={"canonical_smiles": "SMILES"})
    df = df[df["Input_SMILES"].astype(str).ne("nan") & df["SMILES"].astype(str).ne("nan")].copy()
    return df



def build_mode_map(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    mapping: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        inp = str(row["Input_SMILES"])
        out = str(row["SMILES"])
        if inp not in mapping and parse_smiles(out) is not None:
            mapping[inp] = row.to_dict()
    return mapping



def collect_category_rows(
    cl_df: pd.DataFrame,
    plain_map: Dict[str, Dict[str, object]],
    rl_map: Dict[str, Dict[str, object]],
    cl_map: Dict[str, Dict[str, object]],
):
    work = cl_df.copy()
    work["in_scaf"] = work["Input_SMILES"].map(murcko_scaffold_smiles)
    work["out_scaf"] = work["SMILES"].map(murcko_scaffold_smiles)
    work = work[work["in_scaf"].notna() & work["out_scaf"].notna()].copy()
    work["preserved"] = work["in_scaf"] == work["out_scaf"]

    def available_all_modes(inp: str) -> bool:
        return inp in plain_map and inp in rl_map and inp in cl_map and parse_smiles(inp) is not None

    preserved_inputs: List[str] = []
    non_preserved_inputs: List[str] = []
    bio_inputs: List[str] = []

    for _, row in work[work["preserved"]].iterrows():
        inp = str(row["Input_SMILES"])
        if available_all_modes(inp):
            preserved_inputs.append(inp)

    for _, row in work[~work["preserved"]].iterrows():
        inp = str(row["Input_SMILES"])
        if available_all_modes(inp):
            non_preserved_inputs.append(inp)

    bio = work[work["Input_SMILES"].map(has_nitro) & ~work["SMILES"].map(has_nitro)]
    for _, row in bio.iterrows():
        inp = str(row["Input_SMILES"])
        if available_all_modes(inp):
            bio_inputs.append(inp)

    # Keep stable unique order
    preserved_inputs = list(dict.fromkeys(preserved_inputs))
    non_preserved_inputs = list(dict.fromkeys(non_preserved_inputs))
    bio_inputs = list(dict.fromkeys(bio_inputs))

    return preserved_inputs, non_preserved_inputs, bio_inputs



def render_panel(smiles: str, title: str, row: Optional[dict] = None, panel_size=(330, 430)) -> Image.Image:
    panel = Image.new("RGB", panel_size, "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    mol = parse_smiles(smiles)
    if mol is not None:
        img = safe_render_molecule_image(smiles, 300, 250)
        panel.paste(img, (15, 45))
    else:
        draw.text((15, 160), "Invalid SMILES", fill="black", font=font)

    draw.text((10, 10), title, fill="black", font=font)
    for i, line in enumerate(metric_lines(smiles, row)):
        draw.text((10, 342 + i * 16), line, fill="black", font=font)
    draw.rectangle([0, 0, panel_size[0] - 1, panel_size[1] - 1], outline=(175, 175, 175))
    return panel



def build_frame(
    title: str,
    idx: int,
    total: int,
    inp: str,
    plain_row: Dict[str, object],
    rl_row: Dict[str, object],
    cl_row: Dict[str, object],
) -> Image.Image:
    W, H = 1460, 610
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((20, 15), title, fill="black", font=font)
    draw.text((20, 36), f"Compound {idx + 1}/{total}", fill=(80, 80, 80), font=font)

    stages = [
        ("Input", inp, {"SMILES": inp}),
        ("Plain", str(plain_row["SMILES"]), plain_row),
        ("Reinforcement", str(rl_row["SMILES"]), rl_row),
        ("Curriculum", str(cl_row["SMILES"]), cl_row),
    ]

    for i, (label, smi, row) in enumerate(stages):
        x = 20 + i * 355
        y = 70
        panel = render_panel(smi, label, row=row)
        canvas.paste(panel, (x, y))

    return canvas



def save_chunked_gifs(
    title: str,
    inputs: List[str],
    plain_map: Dict[str, Dict[str, object]],
    rl_map: Dict[str, Dict[str, object]],
    cl_map: Dict[str, Dict[str, object]],
    out_prefix: str,
):
    if not inputs:
        print(f"No inputs for {out_prefix}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(inputs)
    part = 1

    for start in range(0, total, PART_SIZE):
        end = min(start + PART_SIZE, total)
        chunk_inputs = inputs[start:end]

        frames: List[Image.Image] = []
        for local_i, inp in enumerate(chunk_inputs):
            global_i = start + local_i
            frame = build_frame(
                title=title,
                idx=global_i,
                total=total,
                inp=inp,
                plain_row=plain_map[inp],
                rl_row=rl_map[inp],
                cl_row=cl_map[inp],
            )
            frames.append(frame)

        out_name = f"{out_prefix}_part_{part:02d}.gif"
        out_path = OUT_DIR / out_name
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=[450] * (len(frames) - 1) + [1200],
            loop=0,
            optimize=True,
        )
        print(f"Created {out_name} ({start + 1}-{end} of {total})")
        part += 1



def main():
    plain_df = load_df(PLAIN_CSV)
    rl_df = load_df(RL_CSV)
    cl_df = load_df(CL_CSV)

    plain_map = build_mode_map(plain_df)
    rl_map = build_mode_map(rl_df)
    cl_map = build_mode_map(cl_df)

    preserved, non_preserved, bio = collect_category_rows(cl_df, plain_map, rl_map, cl_map)

    save_chunked_gifs(
        title="Spiral Preservation (All compounds): Input -> Plain -> RL -> CL",
        inputs=preserved,
        plain_map=plain_map,
        rl_map=rl_map,
        cl_map=cl_map,
        out_prefix="spiral_preservation_all_compounds",
    )

    save_chunked_gifs(
        title="Non-Spiral Preservation (All compounds): Input -> Plain -> RL -> CL",
        inputs=non_preserved,
        plain_map=plain_map,
        rl_map=rl_map,
        cl_map=cl_map,
        out_prefix="non_spiral_preservation_all_compounds",
    )

    save_chunked_gifs(
        title="Bioisostere Replacement (All compounds): Input -> Plain -> RL -> CL",
        inputs=bio,
        plain_map=plain_map,
        rl_map=rl_map,
        cl_map=cl_map,
        out_prefix="bioisostere_replacement_all_compounds",
    )


if __name__ == "__main__":
    main()
