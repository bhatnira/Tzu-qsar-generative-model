from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "generated" / "animations" / "final_structures"

TARGETS = {
    "mol2mol": {
        "plain": ROOT / "outputs" / "generated" / "mol2mol" / "plain" / "mol2mol_plain_generated_optimal.xlsx",
        "reinforcement_learning": ROOT / "outputs" / "generated" / "mol2mol" / "reinforcement_learning" / "mol2mol_rl_generated_optimal.xlsx",
        "curriculum_learning": ROOT / "outputs" / "generated" / "mol2mol" / "curriculum_learning" / "mol2mol_cl_generated_optimal.xlsx",
    },
    "linkinvent": {
        "plain": ROOT / "outputs" / "generated" / "linkinvent" / "plain" / "linkinvent_plain_generated_optimal.csv",
        "reinforcement_learning": ROOT / "outputs" / "generated" / "linkinvent" / "reinforcement_learning" / "linkinvent_rl_generated_optimal.csv",
        "curriculum_learning": ROOT / "outputs" / "generated" / "linkinvent" / "curriculum_learning" / "linkinvent_cl_generated_optimal.csv",
    },
    "linkinvent_transformer_pubchem": {
        "plain": ROOT / "outputs" / "generated" / "linkinvent_transformer_pubchem" / "plain" / "linkinvent_transformer_pubchem_plain_generated_optimal.csv",
        "reinforcement_learning": ROOT / "outputs" / "generated" / "linkinvent_transformer_pubchem" / "reinforcement_learning" / "linkinvent_transformer_pubchem_rl_generated_optimal.csv",
        "curriculum_learning": ROOT / "outputs" / "generated" / "linkinvent_transformer_pubchem" / "curriculum_learning" / "linkinvent_transformer_pubchem_cl_generated_optimal.csv",
    },
    "libinvent": {
        "plain": ROOT / "outputs" / "generated" / "libinvent" / "plain" / "libinvent_plain_generated_optimal.csv",
        "reinforcement_learning": ROOT / "outputs" / "generated" / "libinvent" / "reinforcement_learning" / "libinvent_rl_generated_optimal.csv",
        "curriculum_learning": ROOT / "outputs" / "generated" / "libinvent" / "curriculum_learning" / "libinvent_cl_generated_optimal.csv",
    },
    "libinvent_transformer_pubchem": {
        "plain": ROOT / "outputs" / "generated" / "libinvent_transformer_pubchem" / "plain" / "libinvent_transformer_pubchem_plain_generated_optimal.csv",
        "reinforcement_learning": ROOT / "outputs" / "generated" / "libinvent_transformer_pubchem" / "reinforcement_learning" / "libinvent_transformer_pubchem_rl_generated_optimal.csv",
        "curriculum_learning": ROOT / "outputs" / "generated" / "libinvent_transformer_pubchem" / "curriculum_learning" / "libinvent_transformer_pubchem_cl_generated_optimal.csv",
    },
}

WIDTH, HEIGHT = 1380, 760
BG = (9, 12, 20)
PANEL = (19, 25, 37)
CARD = (28, 36, 52)
TEXT = (237, 240, 245)
MUTED = (148, 163, 184)
ACCENT = (96, 165, 250)
PLAIN = (34, 197, 94)
RL = (251, 191, 36)
CL = (167, 139, 250)
STAGE_COLOR = {
    "plain": PLAIN,
    "reinforcement_learning": RL,
    "curriculum_learning": CL,
}


@dataclass
class StagePick:
    stage: str
    smiles: str
    similarity: float


def _font(size: int, bold: bool = False):
    paths = []
    if bold:
        paths += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ]
    paths += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_TITLE = _font(34, True)
FONT_SUB = _font(19)
FONT_BODY = _font(17)
FONT_BODY_B = _font(18, True)
FONT_SMALL = _font(14)

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def get_smiles_col(df: pd.DataFrame) -> str:
    for col in ["canonical_smiles", "SMILES", "smiles"]:
        if col in df.columns:
            return col
    raise ValueError("No SMILES column found")


def clean_smiles(series: pd.Series) -> list[str]:
    out = []
    seen = set()
    for v in series.astype(str).tolist():
        if not v or v.lower() == "nan":
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def looks_too_complex(smiles: str) -> bool:
    return len(smiles) > 220


def best_final_smiles(df: pd.DataFrame, smiles_col: str) -> str:
    if "pred_pIC50" in df.columns:
        valid = df[df[smiles_col].notna()].copy()
        valid = valid.sort_values("pred_pIC50", ascending=False)
        for smi in valid[smiles_col].astype(str).tolist():
            if smi and smi.lower() != "nan":
                return smi
    for smi in df[smiles_col].astype(str).tolist():
        if smi and smi.lower() != "nan":
            return smi
    raise ValueError("No valid final smiles")


def similarity(a: str, b: str) -> float:
    return 1.0 if a == b else 0.0


def pick_stage_smiles(stage_df: pd.DataFrame, final_smiles: str) -> tuple[str, float]:
    col = get_smiles_col(stage_df)
    candidates = clean_smiles(stage_df[col])
    if not candidates:
        return final_smiles, 0.0
    if final_smiles in candidates:
        return final_smiles, 1.0
    best = candidates[0]
    return best, similarity(best, final_smiles)


def mol_img(smiles: str, size=(320, 220)) -> Image.Image:
    if looks_too_complex(smiles):
        img = Image.new("RGB", size, (250, 250, 250))
        d = ImageDraw.Draw(img)
        d.text((16, 92), "Structure too complex", fill=(40, 40, 40), font=FONT_BODY_B)
        d.text((16, 118), "shown as text below", fill=(70, 70, 70), font=FONT_BODY)
        return img
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return Image.new("RGB", size, (255, 255, 255))
        return Draw.MolToImage(mol, size=size)
    except Exception:
        img = Image.new("RGB", size, (250, 250, 250))
        d = ImageDraw.Draw(img)
        d.text((16, 104), "Could not render", fill=(40, 40, 40), font=FONT_BODY_B)
        return img


def trunc(s: str, n: int = 58) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def draw_card(base: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, pick: StagePick, active: bool):
    color = STAGE_COLOR[pick.stage]
    edge = color if active else (80, 90, 110)
    draw.rounded_rectangle((x, y, x + 404, y + 470), radius=22, fill=CARD, outline=edge, width=5)

    label = pick.stage.replace("_", " ")
    lw = int(draw.textlength(label, font=FONT_SMALL)) + 22
    draw.rounded_rectangle((x + 18, y + 18, x + 18 + lw, y + 46), radius=12, fill=color)
    draw.text((x + 29, y + 25), label, font=FONT_SMALL, fill=(10, 10, 10))

    sim_text = f"similarity to final: {pick.similarity:.3f}"
    draw.text((x + 18, y + 58), sim_text, font=FONT_BODY, fill=MUTED)

    mimg = mol_img(pick.smiles)
    base.paste(mimg, (x + 42, y + 92))

    draw.text((x + 18, y + 332), "Structure", font=FONT_BODY_B, fill=TEXT)
    draw.text((x + 18, y + 360), trunc(pick.smiles, 66), font=FONT_BODY, fill=TEXT)


def make_frame(model: str, picks: list[StagePick], frame: int) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((26, 22, WIDTH - 26, 132), radius=24, fill=PANEL)
    draw.rounded_rectangle((26, 150, WIDTH - 26, HEIGHT - 26), radius=24, fill=PANEL)

    draw.text((50, 44), f"{model}: final structure path", font=FONT_TITLE, fill=TEXT)
    draw.text((52, 92), "Shows only the final structure progression (plain → RL → CL), not all generated molecules.", font=FONT_SUB, fill=MUTED)

    xs = [50, 488, 926]
    y = 220
    for i, pick in enumerate(picks):
        draw_card(img, draw, xs[i], y, pick, active=(i == frame))

    # arrows
    for ax in [458, 896]:
        draw.line((ax, 448, ax + 28, 448), fill=ACCENT, width=5)
        draw.polygon([(ax + 28, 448), (ax + 18, 442), (ax + 18, 454)], fill=ACCENT)

    caption = [
        "Frame 1: closest precursor in plain stage",
        "Frame 2: closest precursor after reinforcement learning",
        "Frame 3: final curriculum-learning structure",
    ][frame]
    draw.text((52, HEIGHT - 58), caption, font=FONT_BODY_B, fill=ACCENT)

    return img


def build_model_gif(model: str, files: dict[str, Path]) -> Optional[Path]:
    try:
        dfs = {stage: load_df(path) for stage, path in files.items()}
        cl_col = get_smiles_col(dfs["curriculum_learning"])
        final = best_final_smiles(dfs["curriculum_learning"], cl_col)

        plain_smi, plain_sim = pick_stage_smiles(dfs["plain"], final)
        rl_smi, rl_sim = pick_stage_smiles(dfs["reinforcement_learning"], final)

        picks = [
            StagePick("plain", plain_smi, plain_sim),
            StagePick("reinforcement_learning", rl_smi, rl_sim),
            StagePick("curriculum_learning", final, 1.0),
        ]

        frames = [make_frame(model, picks, i) for i in range(3)]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{model}_final_structure_path.gif"
        frames[0].save(
            out,
            save_all=True,
            append_images=frames[1:],
            duration=[1400, 1400, 1800],
            loop=0,
            optimize=False,
        )
        return out
    except Exception as exc:
        print(f"SKIP {model}: {exc}")
        return None


def main():
    generated = []
    for model, files in TARGETS.items():
        out = build_model_gif(model, files)
        if out:
            generated.append(out)
            print(out.relative_to(ROOT).as_posix())
    print(f"Generated {len(generated)} focused final-structure GIFs in {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
