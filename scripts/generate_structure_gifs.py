from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "generated" / "animations" / "structures"

TARGETS: dict[str, list[tuple[str, Path]]] = {
    "mol2mol": [
        ("plain", ROOT / "reinvent_integration" / "results" / "mol2mol_plain" / "samples.csv"),
        ("reinforcement_learning", ROOT / "reinvent_integration" / "results" / "mol2mol_rl_adme" / "samples.csv"),
        ("curriculum_learning", ROOT / "reinvent_integration" / "results" / "mol2mol_cl_adme" / "samples.csv"),
    ],
    "linkinvent": [
        ("plain", ROOT / "reinvent_integration" / "results" / "linkinvent_plain" / "samples.csv"),
        ("reinforcement_learning", ROOT / "reinvent_integration" / "results" / "linkinvent_rl" / "samples.csv"),
        ("curriculum_learning", ROOT / "reinvent_integration" / "results" / "linkinvent_cl" / "samples.csv"),
    ],
    "linkinvent_transformer_pubchem": [
        ("plain", ROOT / "reinvent_integration" / "results" / "linkinvent_transformer_pubchem_plain" / "samples.csv"),
        ("reinforcement_learning", ROOT / "reinvent_integration" / "results" / "linkinvent_transformer_pubchem_rl" / "samples.csv"),
        ("curriculum_learning", ROOT / "reinvent_integration" / "results" / "linkinvent_transformer_pubchem_cl" / "samples.csv"),
    ],
    "libinvent": [
        ("plain", ROOT / "reinvent_integration" / "results" / "libinvent_plain" / "samples.csv"),
        ("reinforcement_learning", ROOT / "reinvent_integration" / "results" / "libinvent_rl" / "samples.csv"),
        ("curriculum_learning", ROOT / "reinvent_integration" / "results" / "libinvent_cl" / "samples.csv"),
    ],
    "libinvent_transformer_pubchem": [
        ("plain", ROOT / "reinvent_integration" / "results" / "libinvent_transformer_pubchem_plain" / "samples.csv"),
        ("reinforcement_learning", ROOT / "reinvent_integration" / "results" / "libinvent_transformer_pubchem_rl" / "samples.csv"),
        ("curriculum_learning", ROOT / "reinvent_integration" / "results" / "libinvent_transformer_pubchem_cl" / "samples.csv"),
    ],
}

WIDTH = 1280
HEIGHT = 920
BG = (9, 12, 20)
PANEL = (19, 25, 37)
CARD = (28, 36, 50)
TEXT = (237, 240, 245)
MUTED = (148, 163, 184)
ACCENT = (96, 165, 250)
STAGE_COLORS = {
    "plain": (34, 197, 94),
    "reinforcement_learning": (251, 191, 36),
    "curriculum_learning": (167, 139, 250),
}

CARDS_PER_STAGE = 4
CARD_W = 278
CARD_H = 255
GRID_LEFT = 52
GRID_TOP = 210
COL_GAP = 18
ROW_GAP = 18


@dataclass
class MoleculeEvent:
    stage: str
    smiles: str
    order: int


def font(size: int, bold: bool = False):
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
            pass
    return ImageFont.load_default()


FONT_TITLE = font(34, True)
FONT_SUB = font(20)
FONT_BODY = font(17)
FONT_BODY_BOLD = font(18, True)
FONT_SMALL = font(14)


def truncate(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def load_events(stages: list[tuple[str, Path]]) -> list[MoleculeEvent]:
    events: list[MoleculeEvent] = []
    for stage, path in stages:
        df = pd.read_csv(path)
        smiles_col = "SMILES" if "SMILES" in df.columns else "canonical_smiles"
        valid = []
        seen = set()
        for smi in df[smiles_col].astype(str).tolist():
            if smi in seen:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            seen.add(smi)
            valid.append(smi)
            if len(valid) >= CARDS_PER_STAGE:
                break
        for i, smi in enumerate(valid, start=1):
            events.append(MoleculeEvent(stage=stage, smiles=smi, order=i))
    return events


def mol_image(smiles: str, size=(220, 150)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Image.new("RGB", size, (255, 255, 255))
    return Draw.MolToImage(mol, size=size)


def draw_card(img: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, event: MoleculeEvent, active: bool):
    border = STAGE_COLORS[event.stage] if active else (75, 85, 99)
    draw.rounded_rectangle((x, y, x + CARD_W, y + CARD_H), radius=18, fill=CARD, outline=border, width=4)
    tag = event.stage.replace("_", " ")
    tag_w = int(draw.textlength(tag, font=FONT_SMALL)) + 20
    draw.rounded_rectangle((x + 14, y + 14, x + 14 + tag_w, y + 40), radius=12, fill=border)
    draw.text((x + 24, y + 21), tag, font=FONT_SMALL, fill=(12, 12, 12))
    draw.text((x + CARD_W - 58, y + 18), f"#{event.order}", font=FONT_SMALL, fill=MUTED)

    mimg = mol_image(event.smiles)
    img.paste(mimg, (x + 28, y + 52))

    lines = wrap(draw, truncate(event.smiles, 110), FONT_BODY, CARD_W - 32)
    text_y = y + 208
    for line in lines[:2]:
        draw.text((x + 16, text_y), line, font=FONT_BODY, fill=TEXT)
        text_y += 20


def make_frame(name: str, events: list[MoleculeEvent], idx: int) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((28, 24, WIDTH - 28, 130), radius=22, fill=PANEL)
    draw.rounded_rectangle((28, 150, WIDTH - 28, HEIGHT - 28), radius=22, fill=PANEL)

    draw.text((50, 42), f"{name} structure generation", font=FONT_TITLE, fill=TEXT)
    draw.text((52, 90), "Animated molecule appearance across plain, reinforcement learning, and curriculum learning", font=FONT_SUB, fill=MUTED)

    total = len(events)
    shown = idx + 1
    current = events[idx]
    badge = f"structure {shown}/{total}"
    bw = int(draw.textlength(badge, font=FONT_SMALL)) + 24
    draw.rounded_rectangle((WIDTH - bw - 46, 46, WIDTH - 46, 76), radius=15, fill=ACCENT)
    draw.text((WIDTH - bw - 34, 54), badge, font=FONT_SMALL, fill=(8, 12, 18))

    draw.text((52, 166), f"Current stage: {current.stage.replace('_', ' ')}", font=FONT_BODY_BOLD, fill=STAGE_COLORS[current.stage])
    draw.text((300, 166), f"Current SMILES: {truncate(current.smiles, 95)}", font=FONT_BODY, fill=TEXT)

    for pos, event in enumerate(events[: shown]):
        col = pos % 4
        row = pos // 4
        x = GRID_LEFT + col * (CARD_W + COL_GAP)
        y = GRID_TOP + row * (CARD_H + ROW_GAP)
        draw_card(img, draw, x, y, event, active=(pos == idx))

    footer = "Each frame adds the next generated structure from the underlying sample CSVs."
    draw.text((52, HEIGHT - 54), footer, font=FONT_SMALL, fill=MUTED)
    return img


def save_gif(name: str, events: list[MoleculeEvent]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [make_frame(name, events, i) for i in range(len(events))]
    out = OUT_DIR / f"{name}_structures.gif"
    durations = [700] * max(1, len(frames) - 1) + [1600]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    return out


def main(names: Iterable[str] | None = None):
    selected = list(names) if names else list(TARGETS)
    made = []
    for name in selected:
        events = load_events(TARGETS[name])
        if not events:
            continue
        out = save_gif(name, events)
        made.append(out)
        print(out.relative_to(ROOT).as_posix())
    print(f"Generated {len(made)} structure GIFs in {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
