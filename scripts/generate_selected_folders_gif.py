#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger

from animation_metrics import load_table, metric_lines, safe_render_molecule_image

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT_GIF = ROOT / "outputs/generated/animations/selected_folders_within.gif"
OUT_PNG = ROOT / "outputs/generated/animations/selected_folders_within_first_frame.png"

MODE_ORDER = ["plain", "reinforcement_learning", "curriculum_learning"]
MODE_LABEL = {
    "plain": "Plain",
    "reinforcement_learning": "RL",
    "curriculum_learning": "CL",
}

TARGETS = [
    ROOT / "outputs/generated/nitro_bioisostere_campaign",
    ROOT / "outputs/generated/manifests",
    ROOT / "outputs/generated/linkinvent_transformer_pubchem",
    ROOT / "outputs/generated/linkinvent",
    ROOT / "outputs/generated/libinvent_transformer_pubchem",
    ROOT / "outputs/generated/libinvent",
]

CSV_PATTERNS = [
    "*_generated_scored.csv",
    "*_generated_optimal.csv",
    "*_generated_scored.xlsx",
    "*_generated_optimal.xlsx",
]


class Row:
    def __init__(self, name: str):
        self.name = name
        self.by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MODE_ORDER}
        self.meta_only: bool = False
        self.meta_lines: List[str] = []



def valid_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or not smiles.strip() or smiles.lower() == "nan":
        return False
    return Chem.MolFromSmiles(smiles) is not None



def smiles_col(columns: List[str]) -> Optional[str]:
    for c in ["canonical_smiles", "SMILES", "smiles"]:
        if c in columns:
            return c
    return None



def load_records(mode_dir: Path, limit: int = 200) -> List[Dict[str, Any]]:
    paths: List[Path] = []
    for pattern in CSV_PATTERNS:
        paths.extend(sorted(mode_dir.glob(pattern)))
    if not paths:
        return []

    try:
        df = load_table(paths[0])
    except Exception:
        return []

    if "SMILES" not in df.columns and "canonical_smiles" in df.columns:
        df = df.rename(columns={"canonical_smiles": "SMILES"})

    col = smiles_col(list(df.columns))
    if col is None:
        return []

    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        smi = str(r[col])
        if valid_smiles(smi):
            d = r.to_dict()
            d["_display_smiles"] = smi
            out.append(d)
        if len(out) >= limit:
            break
    return out



def discover_rows() -> List[Row]:
    rows: List[Row] = []

    # Nitro campaign includes multiple sub-campaign folders
    nitro_root = TARGETS[0]
    if nitro_root.exists():
        for child in sorted(nitro_root.iterdir()):
            if not child.is_dir():
                continue
            if all((child / m).is_dir() for m in MODE_ORDER):
                r = Row(f"nitro/{child.name}")
                for m in MODE_ORDER:
                    r.by_mode[m] = load_records(child / m)
                rows.append(r)

    # manifests folder -> metadata row
    manifest_root = TARGETS[1]
    if manifest_root.exists():
        r = Row("manifests")
        r.meta_only = True
        files = sorted([p.name for p in manifest_root.iterdir() if p.is_file()])
        r.meta_lines = ["Metadata workbook(s):"] + files[:6]
        rows.append(r)

    # remaining direct family folders
    for family_root in TARGETS[2:]:
        if not family_root.exists() or not family_root.is_dir():
            continue
        if all((family_root / m).is_dir() for m in MODE_ORDER):
            r = Row(family_root.name)
            for m in MODE_ORDER:
                r.by_mode[m] = load_records(family_root / m)
            rows.append(r)

    # remove empty non-meta rows
    filtered = []
    for r in rows:
        if r.meta_only:
            filtered.append(r)
            continue
        if sum(len(r.by_mode[m]) for m in MODE_ORDER) > 0:
            filtered.append(r)
    return filtered



def pick_idx(n: int, frame: int, total: int) -> int:
    if n <= 1 or total <= 1:
        return 0
    return min(n - 1, int(round(frame * (n - 1) / (total - 1))))



def draw_panel(record: Optional[Dict[str, Any]], title: str, w: int = 330, h: int = 280) -> Image.Image:
    panel = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill="black", font=font)

    if not record:
        draw.text((12, 130), "No data", fill=(90, 90, 90), font=font)
        draw.rectangle([0, 0, w - 1, h - 1], outline=(170, 170, 170))
        return panel

    smiles = record.get("_display_smiles")
    if isinstance(smiles, str) and valid_smiles(smiles):
        img = safe_render_molecule_image(smiles, w - 16, 150)
        panel.paste(img, (8, 30))
        lines = metric_lines(smiles, record)
        for i, line in enumerate(lines):
            draw.text((8, 188 + i * 16), line, fill="black", font=font)
    else:
        draw.text((12, 130), "Invalid", fill="black", font=font)

    draw.rectangle([0, 0, w - 1, h - 1], outline=(170, 170, 170))
    return panel



def draw_meta_panel(lines: List[str], title: str, w: int = 330, h: int = 280) -> Image.Image:
    panel = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill="black", font=font)
    for i, line in enumerate(lines[:12]):
        draw.text((8, 40 + i * 18), line, fill="black", font=font)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(170, 170, 170))
    return panel



def build_frames(rows: List[Row], total_frames: int = 14) -> List[Image.Image]:
    row_h = 300
    left_w = 240
    panel_w = 330
    panel_h = 280
    gap = 12

    width = left_w + 3 * panel_w + 4 * gap
    height = 92 + len(rows) * row_h + 20

    frames: List[Image.Image] = []
    font = ImageFont.load_default()

    for frame_idx in range(total_frames):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)

        draw.text((14, 12), "Selected Folders Only (within requested paths)", fill="black", font=font)
        draw.text((14, 34), f"Step {frame_idx + 1}/{total_frames}", fill=(80, 80, 80), font=font)

        x0 = left_w + gap
        for i, mode in enumerate(MODE_ORDER):
            draw.text((x0 + i * (panel_w + gap) + 8, 60), MODE_LABEL[mode], fill="black", font=font)

        for r_idx, row in enumerate(rows):
            y = 88 + r_idx * row_h
            draw.text((12, y + 120), row.name, fill="black", font=font)

            if row.meta_only:
                for i, mode in enumerate(MODE_ORDER):
                    panel = draw_meta_panel(row.meta_lines, f"{MODE_LABEL[mode]} (meta)", panel_w, panel_h)
                    x = x0 + i * (panel_w + gap)
                    canvas.paste(panel, (x, y))
                continue

            for i, mode in enumerate(MODE_ORDER):
                records = row.by_mode.get(mode, [])
                idx = pick_idx(len(records), frame_idx, total_frames)
                rec = records[idx] if records else None
                panel = draw_panel(rec, MODE_LABEL[mode], panel_w, panel_h)
                x = x0 + i * (panel_w + gap)
                canvas.paste(panel, (x, y))

        frames.append(canvas)

    return frames



def main() -> None:
    rows = discover_rows()
    if not rows:
        raise RuntimeError("No rows discovered for requested target paths")

    frames = build_frames(rows, total_frames=14)
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=[750] * (len(frames) - 1) + [1500],
        loop=0,
        optimize=True,
    )
    frames[0].save(OUT_PNG)

    print(f"Created: {OUT_GIF}")
    print(f"Created: {OUT_PNG}")
    print(f"Rows included: {len(rows)}")


if __name__ == "__main__":
    main()
