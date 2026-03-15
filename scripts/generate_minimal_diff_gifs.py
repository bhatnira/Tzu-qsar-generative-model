#!/usr/bin/env python3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from animation_metrics import load_table, metric_lines, safe_render_molecule_image

ROOT = Path(__file__).resolve().parents[1]
PLAIN_CSV = ROOT / "outputs/generated/mol2mol/plain/mol2mol_plain_generated_scored.xlsx"
RL_CSV = ROOT / "outputs/generated/mol2mol/reinforcement_learning/mol2mol_rl_generated_scored.xlsx"
CL_CSV = ROOT / "outputs/generated/mol2mol/curriculum_learning/mol2mol_cl_generated_scored.xlsx"
OUT_DIR = ROOT / "outputs/generated/animations/final_structures/minimal_diffs"

NITRO_PATTERN = "[N+](=O)[O-]"


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip() or smiles.lower() == "nan":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
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


def pick_valid_row(df: pd.DataFrame, input_smiles: str) -> Optional[pd.Series]:
    subset = df[df["Input_SMILES"] == input_smiles]
    for _, row in subset.iterrows():
        if parse_smiles(str(row["SMILES"])) is not None:
            return row
    return None


def find_category_examples(
    plain_df: pd.DataFrame, rl_df: pd.DataFrame, cl_df: pd.DataFrame
) -> Tuple[str, str, str]:
    cl_work = cl_df.copy()
    cl_work["in_scaf"] = cl_work["Input_SMILES"].map(murcko_scaffold_smiles)
    cl_work["out_scaf"] = cl_work["SMILES"].map(murcko_scaffold_smiles)
    cl_work = cl_work[cl_work["in_scaf"].notna() & cl_work["out_scaf"].notna()].copy()
    cl_work["preserved"] = cl_work["in_scaf"] == cl_work["out_scaf"]

    preserved_input = None
    nonpreserved_input = None
    bio_input = None

    for _, row in cl_work[cl_work["preserved"]].iterrows():
        candidate = str(row["Input_SMILES"])
        if (
            parse_smiles(candidate) is not None
            and pick_valid_row(plain_df, candidate) is not None
            and pick_valid_row(rl_df, candidate) is not None
            and pick_valid_row(cl_df, candidate) is not None
        ):
            preserved_input = candidate
            break

    for _, row in cl_work[~cl_work["preserved"]].iterrows():
        candidate = str(row["Input_SMILES"])
        if (
            parse_smiles(candidate) is not None
            and pick_valid_row(plain_df, candidate) is not None
            and pick_valid_row(rl_df, candidate) is not None
            and pick_valid_row(cl_df, candidate) is not None
        ):
            nonpreserved_input = candidate
            break

    bio_candidates = cl_work[
        cl_work["Input_SMILES"].map(has_nitro) & ~cl_work["SMILES"].map(has_nitro)
    ]
    for _, row in bio_candidates.iterrows():
        candidate = str(row["Input_SMILES"])
        if (
            parse_smiles(candidate) is not None
            and pick_valid_row(plain_df, candidate) is not None
            and pick_valid_row(rl_df, candidate) is not None
            and pick_valid_row(cl_df, candidate) is not None
        ):
            bio_input = candidate
            break

    if not preserved_input or not nonpreserved_input or not bio_input:
        raise RuntimeError("Could not find robust examples for all categories")

    return preserved_input, nonpreserved_input, bio_input


def render_mol_panel(smiles: str, title: str, row: Optional[dict] = None, panel_size=(320, 420)) -> Image.Image:
    panel = Image.new("RGB", panel_size, "white")
    draw = ImageDraw.Draw(panel)

    mol = parse_smiles(smiles)
    if mol is not None:
        mol_img = safe_render_molecule_image(smiles, 280, 250)
        panel.paste(mol_img, (20, 40))
    else:
        draw.text((20, 140), "Invalid SMILES", fill="black")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((12, 12), title, fill="black", font=font)
    for i, line in enumerate(metric_lines(smiles, row)):
        draw.text((12, 332 + i * 16), line, fill="black", font=font)
    return panel


def create_comparison_gif(
    input_row: dict,
    plain_row: dict,
    rl_row: dict,
    cl_row: dict,
    category_title: str,
    output_path: Path,
):
    canvas_size = (1400, 600)
    frames = []
    stage_data = [
        ("Input", str(input_row["SMILES"]), input_row),
        ("Plain", str(plain_row["SMILES"]), plain_row),
        ("RL", str(rl_row["SMILES"]), rl_row),
        ("CL", str(cl_row["SMILES"]), cl_row),
    ]

    for highlight_idx in range(len(stage_data)):
        canvas = Image.new("RGB", canvas_size, "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw.text((30, 18), category_title, fill="black", font=font)

        x0 = 25
        for idx, (label, smiles, row) in enumerate(stage_data):
            panel = render_mol_panel(smiles, label, row=row)
            px = x0 + idx * 340
            py = 80
            canvas.paste(panel, (px, py))

            border_color = "black" if idx == highlight_idx else (190, 190, 190)
            border_width = 3 if idx == highlight_idx else 1
            for w in range(border_width):
                draw.rectangle(
                    [px - w, py - w, px + 320 + w, py + 420 + w],
                    outline=border_color,
                )

        frames.append(canvas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[1200, 1200, 1200, 1600],
        loop=0,
        optimize=True,
    )


def build_gif_for_input(
    category_title: str,
    input_smiles: str,
    plain_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    cl_df: pd.DataFrame,
    output_name: str,
):
    plain_row = pick_valid_row(plain_df, input_smiles)
    rl_row = pick_valid_row(rl_df, input_smiles)
    cl_row = pick_valid_row(cl_df, input_smiles)

    if plain_row is None or rl_row is None or cl_row is None:
        raise RuntimeError(f"Missing valid rows for input: {input_smiles}")

    input_row = {"SMILES": input_smiles}
    create_comparison_gif(
        input_row=input_row,
        plain_row=plain_row.to_dict(),
        rl_row=rl_row.to_dict(),
        cl_row=cl_row.to_dict(),
        category_title=category_title,
        output_path=OUT_DIR / output_name,
    )


def main():
    plain_df = load_df(PLAIN_CSV)
    rl_df = load_df(RL_CSV)
    cl_df = load_df(CL_CSV)

    preserved_input, nonpreserved_input, bio_input = find_category_examples(
        plain_df, rl_df, cl_df
    )

    build_gif_for_input(
        category_title="Scaffold Preservation: Input vs Plain vs RL vs CL",
        input_smiles=preserved_input,
        plain_df=plain_df,
        rl_df=rl_df,
        cl_df=cl_df,
        output_name="scaffold_preservation_minimal.gif",
    )

    build_gif_for_input(
        category_title="Scaffold Non-Preservation: Input vs Plain vs RL vs CL",
        input_smiles=nonpreserved_input,
        plain_df=plain_df,
        rl_df=rl_df,
        cl_df=cl_df,
        output_name="scaffold_non_preservation_minimal.gif",
    )

    build_gif_for_input(
        category_title="Bioisostere-like Replacement: Input vs Plain vs RL vs CL",
        input_smiles=bio_input,
        plain_df=plain_df,
        rl_df=rl_df,
        cl_df=cl_df,
        output_name="bioisostere_replacement_minimal.gif",
    )

    print("Created minimal comparison GIFs in:", OUT_DIR)
    print("- scaffold_preservation_minimal.gif")
    print("- scaffold_non_preservation_minimal.gif")
    print("- bioisostere_replacement_minimal.gif")


if __name__ == "__main__":
    main()
