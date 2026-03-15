#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger

from animation_metrics import load_table, metric_lines, safe_render_molecule_image

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs/generated"
OUT_DIR = BASE_DIR / "animations"
OUT_GIF = OUT_DIR / "all_generation_folders_big_timeline.gif"
OUT_PNG = OUT_DIR / "all_generation_folders_big_timeline_first_frame.png"

MODE_MAP = {
    "plain": "Plain",
    "reinforcement_learning": "RL",
    "curriculum_learning": "CL",
}
MODE_ORDER = ["plain", "reinforcement_learning", "curriculum_learning"]
CSV_CANDIDATES = [
    "*_generated_scored.csv",
    "*_generated_optimal.csv",
]


class FolderData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.records_by_mode: Dict[str, List[Dict[str, Any]]] = {k: [] for k in MODE_ORDER}



def valid_smiles(value: str) -> bool:
    if not isinstance(value, str) or not value.strip() or value.lower() == "nan":
        return False
    mol = Chem.MolFromSmiles(value)
    return mol is not None



def choose_smiles_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["canonical_smiles", "SMILES", "smiles"]:
        if candidate in df.columns:
            return candidate
    return None



def load_mode_records(mode_dir: Path, limit: int = 200) -> List[Dict[str, Any]]:
    candidates: List[Path] = []
    for pattern in CSV_CANDIDATES:
        candidates.extend(sorted(mode_dir.glob(pattern)))
    candidates.extend(sorted(mode_dir.glob("*_generated_scored.xlsx")))
    candidates.extend(sorted(mode_dir.glob("*_generated_optimal.xlsx")))
    if not candidates:
        return []
    table_path = candidates[0]

    try:
        df = load_table(table_path)
    except Exception:
        return []

    col = choose_smiles_column(df)
    if col is None:
        return []

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        value = str(row[col])
        if valid_smiles(value):
            rec = row.to_dict()
            rec["_display_smiles"] = value
            records.append(rec)
        if len(records) >= limit:
            break
    return records



def discover_generation_folders(base_dir: Path) -> List[FolderData]:
    folders: List[FolderData] = []

    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"animations", "manifests", "qsar", "adme_final", "adme_legacy"}:
            continue

        has_mode_subdirs = all((child / mode).is_dir() for mode in MODE_ORDER)
        if not has_mode_subdirs:
            continue

        folder_data = FolderData(child.name)
        total = 0
        for mode in MODE_ORDER:
            records = load_mode_records(child / mode)
            folder_data.records_by_mode[mode] = records
            total += len(records)

        if total > 0:
            folders.append(folder_data)

    return folders



def pick_index(seq_len: int, frame_idx: int, total_frames: int) -> int:
    if seq_len <= 1:
        return 0
    if total_frames <= 1:
        return 0
    return min(seq_len - 1, int(round(frame_idx * (seq_len - 1) / (total_frames - 1))))



def draw_molecule_panel(record: Optional[Dict[str, Any]], title: str, size: Tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    smiles = record.get("_display_smiles") if record else None

    if smiles and valid_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol_img = safe_render_molecule_image(smiles, size[0] - 20, size[1] - 118)
            panel.paste(mol_img, (10, 28))
        else:
            draw.text((14, 55), "Invalid", fill="black", font=font)
    else:
        draw.text((14, 55), "No data", fill=(80, 80, 80), font=font)

    draw.text((10, 8), title, fill="black", font=font)
    if smiles:
        for i, line in enumerate(metric_lines(smiles, record)):
            draw.text((10, size[1] - 92 + i * 16), line, fill="black", font=font)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(170, 170, 170))
    return panel



def build_frames(folders: List[FolderData], total_frames: int = 10) -> List[Image.Image]:
    row_h = 300
    left_w = 220
    panel_w = 330
    panel_h = 280
    gap = 12

    width = left_w + 3 * panel_w + 4 * gap
    height = 90 + len(folders) * row_h + 20

    frames: List[Image.Image] = []
    font = ImageFont.load_default()

    for frame_idx in range(total_frames):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)

        draw.text((14, 12), "All Generation Folders: Plain vs RL vs CL (time-evolving)", fill="black", font=font)
        draw.text((14, 36), f"Step {frame_idx + 1}/{total_frames}", fill=(70, 70, 70), font=font)

        x_base = left_w + gap
        for mode_i, mode in enumerate(MODE_ORDER):
            x = x_base + mode_i * (panel_w + gap)
            draw.text((x + 6, 62), MODE_MAP[mode], fill="black", font=font)

        for row, folder in enumerate(folders):
            y = 88 + row * row_h
            draw.text((14, y + 118), folder.name, fill="black", font=font)

            for mode_i, mode in enumerate(MODE_ORDER):
                mode_records = folder.records_by_mode.get(mode, [])
                idx = pick_index(len(mode_records), frame_idx, total_frames)
                record = mode_records[idx] if mode_records else None

                panel = draw_molecule_panel(record, MODE_MAP[mode], (panel_w, panel_h))
                x = x_base + mode_i * (panel_w + gap)
                canvas.paste(panel, (x, y))

                if mode_i == (frame_idx % 3):
                    for w in range(2):
                        draw.rectangle([x - w, y - w, x + panel_w + w, y + panel_h + w], outline="black")

        frames.append(canvas)

    return frames



def main() -> None:
    folders = discover_generation_folders(BASE_DIR)
    if not folders:
        raise RuntimeError("No generation folders with plain/RL/CL data were found.")

    frames = build_frames(folders, total_frames=10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=[1000] * (len(frames) - 1) + [1800],
        loop=0,
        optimize=True,
    )
    frames[0].save(OUT_PNG)

    print(f"Folders included: {len(folders)}")
    print(f"Created GIF: {OUT_GIF}")
    print(f"Created first-frame PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
