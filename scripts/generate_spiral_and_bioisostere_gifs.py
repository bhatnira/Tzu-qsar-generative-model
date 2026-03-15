#!/usr/bin/env python3
from pathlib import Path
from typing import Optional, Tuple

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



def load_df(path: Path) -> pd.DataFrame:
    df = load_table(path)
    if "SMILES" not in df.columns and "canonical_smiles" in df.columns:
        df = df.rename(columns={"canonical_smiles": "SMILES"})
    df = df[df["Input_SMILES"].astype(str).ne("nan") & df["SMILES"].astype(str).ne("nan")].copy()
    return df



def has_nitro(smiles: str) -> bool:
    return isinstance(smiles, str) and NITRO_PATTERN in smiles



def pick_valid_row(df: pd.DataFrame, input_smiles: str) -> Optional[pd.Series]:
    subset = df[df["Input_SMILES"] == input_smiles]
    for _, row in subset.iterrows():
        if parse_smiles(str(row["SMILES"])) is not None:
            return row
    return None



def find_examples(plain_df: pd.DataFrame, rl_df: pd.DataFrame, cl_df: pd.DataFrame) -> Tuple[str, str, str]:
    cl = cl_df.copy()
    cl["in_scaf"] = cl["Input_SMILES"].map(murcko_scaffold_smiles)
    cl["out_scaf"] = cl["SMILES"].map(murcko_scaffold_smiles)
    cl = cl[cl["in_scaf"].notna() & cl["out_scaf"].notna()].copy()
    cl["preserved"] = cl["in_scaf"] == cl["out_scaf"]

    def valid_all_modes(input_smi: str) -> bool:
        return (
            parse_smiles(input_smi) is not None
            and pick_valid_row(plain_df, input_smi) is not None
            and pick_valid_row(rl_df, input_smi) is not None
            and pick_valid_row(cl_df, input_smi) is not None
        )

    preserved_input = None
    for _, row in cl[cl["preserved"]].iterrows():
        s = str(row["Input_SMILES"])
        if valid_all_modes(s):
            preserved_input = s
            break

    non_preserved_input = None
    for _, row in cl[~cl["preserved"]].iterrows():
        s = str(row["Input_SMILES"])
        if valid_all_modes(s):
            non_preserved_input = s
            break

    bio_input = None
    bio = cl[cl["Input_SMILES"].map(has_nitro) & ~cl["SMILES"].map(has_nitro)]
    for _, row in bio.iterrows():
        s = str(row["Input_SMILES"])
        if valid_all_modes(s):
            bio_input = s
            break

    if not preserved_input or not non_preserved_input or not bio_input:
        raise RuntimeError("Could not find all required examples")

    return preserved_input, non_preserved_input, bio_input



def render_panel(smiles: str, title: str, row: Optional[dict] = None, panel_size=(380, 480)) -> Image.Image:
    panel = Image.new("RGB", panel_size, "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    mol = parse_smiles(smiles)
    if mol is not None:
        mol_img = safe_render_molecule_image(smiles, 340, 300)
        panel.paste(mol_img, (20, 50))
    else:
        draw.text((20, 180), "Invalid SMILES", fill="black", font=font)

    draw.text((12, 12), title, fill="black", font=font)
    for i, line in enumerate(metric_lines(smiles, row)):
        draw.text((12, 392 + i * 16), line, fill="black", font=font)
    return panel



def create_big_gif(input_row: dict, plain_row: dict, rl_row: dict, cl_row: dict, title: str, out_path: Path):
    stage_data = [
        ("Input", str(input_row["SMILES"]), input_row),
        ("Plain", str(plain_row["SMILES"]), plain_row),
        ("Reinforcement", str(rl_row["SMILES"]), rl_row),
        ("Curriculum", str(cl_row["SMILES"]), cl_row),
    ]

    frames = []
    W, H = 1700, 720
    font = ImageFont.load_default()

    for hi in range(4):
        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 18), title, fill="black", font=font)

        for i, (label, smi, row) in enumerate(stage_data):
            x = 20 + i * 420
            y = 80
            panel = render_panel(smi, label, row=row)
            canvas.paste(panel, (x, y))

            color = "black" if i == hi else (185, 185, 185)
            bw = 4 if i == hi else 1
            for w in range(bw):
                draw.rectangle([x - w, y - w, x + 380 + w, y + 480 + w], outline=color)

        frames.append(canvas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=[1300, 1300, 1300, 1800],
        loop=0,
        optimize=True,
    )



def main():
    plain_df = load_df(PLAIN_CSV)
    rl_df = load_df(RL_CSV)
    cl_df = load_df(CL_CSV)

    preserved_input, non_preserved_input, bio_input = find_examples(plain_df, rl_df, cl_df)

    def get_triplet(input_smi: str):
        pr = pick_valid_row(plain_df, input_smi)
        rr = pick_valid_row(rl_df, input_smi)
        cr = pick_valid_row(cl_df, input_smi)
        if pr is None or rr is None or cr is None:
            raise RuntimeError(f"Missing rows for input: {input_smi}")
        return pr.to_dict(), rr.to_dict(), cr.to_dict()

    p_plain, p_rl, p_cl = get_triplet(preserved_input)
    n_plain, n_rl, n_cl = get_triplet(non_preserved_input)
    b_plain, b_rl, b_cl = get_triplet(bio_input)
    p_input = {"SMILES": preserved_input}
    n_input = {"SMILES": non_preserved_input}
    b_input = {"SMILES": bio_input}

    create_big_gif(
        p_input,
        p_plain,
        p_rl,
        p_cl,
        "Spiral Preservation (scaffold preserved): Input -> Plain -> RL -> CL",
        OUT_DIR / "spiral_preservation_big.gif",
    )

    create_big_gif(
        n_input,
        n_plain,
        n_rl,
        n_cl,
        "Non-Spiral Preservation (scaffold changed): Input -> Plain -> RL -> CL",
        OUT_DIR / "non_spiral_preservation_big.gif",
    )

    create_big_gif(
        b_input,
        b_plain,
        b_rl,
        b_cl,
        "Bioisostere Replacement: Input -> Plain -> RL -> CL",
        OUT_DIR / "bioisostere_replacement_big.gif",
    )

    # compatibility alias with user typo spelling
    create_big_gif(
        b_input,
        b_plain,
        b_rl,
        b_cl,
        "Biostere Replacement: Input -> Plain -> RL -> CL",
        OUT_DIR / "biostere_replacement_big.gif",
    )

    print("Created:")
    print("- spiral_preservation_big.gif")
    print("- non_spiral_preservation_big.gif")
    print("- bioisostere_replacement_big.gif")
    print("- biostere_replacement_big.gif")


if __name__ == "__main__":
    main()
