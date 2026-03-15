#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/generated/animations"
OUT_GIF = OUT_DIR / "all_requested_outputs_combined.gif"
OUT_PNG = OUT_DIR / "all_requested_outputs_combined_first_frame.png"

FILES = [
    ROOT / "outputs/generated/animations/all_generation_folders_big_timeline_first_frame.png",
    ROOT / "outputs/generated/animations/all_generation_folders_big_timeline.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/spiral_preservation_big.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/spiral_preservation_all_compounds_part_01.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/scaffold_preservation_minimal.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/scaffold_non_preservation_minimal.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/non_spiral_preservation_big.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/non_spiral_preservation_all_compounds_part_01.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/minimal_combined_summary.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/biostere_replacement_big.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/bioisostere_replacement_minimal.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/bioisostere_replacement_big.gif",
    ROOT / "outputs/generated/animations/final_structures/minimal_diffs/bioisostere_replacement_all_compounds_part_01.gif",
]

TILE_W = 360
TILE_H = 220
PADDING = 16
HEADER_H = 54
COLS = 3
BG = "white"


class Asset:
    def __init__(self, path: Path, frames: List[Image.Image]):
        self.path = path
        self.label = path.name
        self.frames = frames



def load_asset(path: Path) -> Asset:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    frames: List[Image.Image] = []

    if suffix == ".gif":
        with Image.open(path) as img:
            for frame in ImageSequence.Iterator(img):
                frames.append(frame.convert("RGB"))
    else:
        with Image.open(path) as img:
            frames.append(img.convert("RGB"))

    if not frames:
        raise RuntimeError(f"No frames found in {path}")

    return Asset(path, frames)



def fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), BG)
    fitted = img.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = (width - fitted.width) // 2
    y = (height - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas



def make_tile(asset: Asset, frame_idx: int, font: ImageFont.ImageFont) -> Image.Image:
    tile = Image.new("RGB", (TILE_W, TILE_H), BG)
    draw = ImageDraw.Draw(tile)

    src = asset.frames[frame_idx % len(asset.frames)]
    img = fit_image(src, TILE_W - 12, TILE_H - 34)
    tile.paste(img, (6, 28))
    draw.text((8, 8), asset.label, fill="black", font=font)
    draw.rectangle([0, 0, TILE_W - 1, TILE_H - 1], outline=(175, 175, 175))
    return tile



def build_master_frames(assets: List[Asset]) -> List[Image.Image]:
    rows = (len(assets) + COLS - 1) // COLS
    width = PADDING + COLS * (TILE_W + PADDING)
    height = HEADER_H + PADDING + rows * (TILE_H + PADDING)
    font = ImageFont.load_default()

    max_frames = max(len(a.frames) for a in assets)
    total_frames = max(max_frames, 12)

    master_frames: List[Image.Image] = []
    for frame_idx in range(total_frames):
        canvas = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((PADDING, 14), "Combined Animation Board: all requested outputs", fill="black", font=font)
        draw.text((PADDING, 32), f"Frame {frame_idx + 1}/{total_frames}", fill=(80, 80, 80), font=font)

        for i, asset in enumerate(assets):
            row = i // COLS
            col = i % COLS
            x = PADDING + col * (TILE_W + PADDING)
            y = HEADER_H + PADDING + row * (TILE_H + PADDING)
            tile = make_tile(asset, frame_idx, font)
            canvas.paste(tile, (x, y))

        master_frames.append(canvas)

    return master_frames



def main() -> None:
    assets = [load_asset(path) for path in FILES]
    frames = build_master_frames(assets)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=[700] * (len(frames) - 1) + [1400],
        loop=0,
        optimize=True,
    )
    frames[0].save(OUT_PNG)

    print(f"Created: {OUT_GIF}")
    print(f"Created: {OUT_PNG}")
    print(f"Assets combined: {len(assets)}")


if __name__ == "__main__":
    main()
