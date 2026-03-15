#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs/generated/animations/everything_in_one_with_phases.gif"
OUT_PREVIEW = ROOT / "outputs/generated/animations/everything_in_one_with_phases_first_frame.png"


@dataclass
class Phase:
    title: str
    subtitle: str
    path: Path
    hold_ms: int = 1400


PHASES: List[Phase] = [
    Phase(
        title="Phase 1: Global Generation Overview",
        subtitle="All folders, synchronized Plain / RL / CL timeline",
        path=ROOT / "outputs/generated/animations/all_generation_folders_big_timeline.gif",
    ),
    Phase(
        title="Phase 2: Scaffold Preservation (Representative)",
        subtitle="Single preserved-scaffold trajectory across Input / Plain / RL / CL",
        path=ROOT / "outputs/generated/animations/final_structures/minimal_diffs/spiral_preservation_big.gif",
    ),
    Phase(
        title="Phase 3: Scaffold Preservation (All Compounds)",
        subtitle="All preserved-scaffold compound trajectories",
        path=ROOT / "outputs/generated/animations/final_structures/minimal_diffs/spiral_preservation_all_compounds_part_01.gif",
    ),
    Phase(
        title="Phase 4: Scaffold Non-Preservation (Representative)",
        subtitle="Single non-preserved scaffold trajectory",
        path=ROOT / "outputs/generated/animations/final_structures/minimal_diffs/non_spiral_preservation_big.gif",
    ),
    Phase(
        title="Phase 5: Scaffold Non-Preservation (All Compounds)",
        subtitle="All non-preserved-scaffold trajectories",
        path=ROOT / "outputs/generated/animations/final_structures/minimal_diffs/non_spiral_preservation_all_compounds_part_01.gif",
    ),
    Phase(
        title="Phase 6: Bioisostere Replacement (Representative)",
        subtitle="Single bioisostere replacement trajectory",
        path=ROOT / "outputs/generated/animations/final_structures/minimal_diffs/bioisostere_replacement_big.gif",
    ),
    Phase(
        title="Phase 7: Bioisostere Replacement (All Compounds)",
        subtitle="All bioisostere replacement trajectories",
        path=ROOT / "outputs/generated/animations/final_structures/minimal_diffs/bioisostere_replacement_all_compounds_part_01.gif",
    ),
    Phase(
        title="Phase 8: Minimal Category Summary",
        subtitle="Preserved vs Non-preserved vs Bioisostere side-by-side summary",
        path=ROOT / "outputs/generated/animations/final_structures/minimal_diffs/minimal_combined_summary.gif",
    ),
    Phase(
        title="Phase 9: Full Combined Board",
        subtitle="All requested visual outputs on one canvas",
        path=ROOT / "outputs/generated/animations/all_requested_outputs_combined.gif",
    ),
]


def read_gif_frames(path: Path) -> List[Image.Image]:
    frames: List[Image.Image] = []
    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            frames.append(frame.convert("RGB"))
    return frames


def fit_image(img: Image.Image, w: int, h: int) -> Image.Image:
    canvas = Image.new("RGB", (w, h), "white")
    cp = img.copy()
    cp.thumbnail((w, h), Image.Resampling.LANCZOS)
    x = (w - cp.width) // 2
    y = (h - cp.height) // 2
    canvas.paste(cp, (x, y))
    return canvas


def make_header_frame(width: int, height: int, title: str, subtitle: str) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((30, 28), "Everything In One GIF (Phased)", fill="black", font=font)
    draw.text((30, 58), title, fill="black", font=font)
    draw.text((30, 82), subtitle, fill=(60, 60, 60), font=font)
    draw.rectangle([20, 18, width - 20, 110], outline=(180, 180, 180))
    return canvas


def wrap_phase_frames(phase: Phase, target_size=(1600, 950), max_content_frames=24) -> tuple[list[Image.Image], list[int]]:
    raw_frames = read_gif_frames(phase.path)
    if not raw_frames:
        return [], []

    step = max(1, len(raw_frames) // max_content_frames)
    sampled = raw_frames[::step][:max_content_frames]

    out_frames: List[Image.Image] = []
    out_durations: List[int] = []

    header = make_header_frame(target_size[0], target_size[1], phase.title, phase.subtitle)
    out_frames.append(header)
    out_durations.append(1200)

    for frame in sampled:
        canvas = header.copy()
        content = fit_image(frame, target_size[0] - 60, target_size[1] - 150)
        canvas.paste(content, (30, 130))
        out_frames.append(canvas)
        out_durations.append(260)

    out_frames.append(out_frames[-1].copy())
    out_durations.append(phase.hold_ms)
    return out_frames, out_durations


def main() -> None:
    missing = [p.path for p in PHASES if not p.path.exists()]
    if missing:
        raise FileNotFoundError("Missing input(s):\n" + "\n".join(str(m) for m in missing))

    all_frames: List[Image.Image] = []
    all_durations: List[int] = []

    for phase in PHASES:
        frames, durations = wrap_phase_frames(phase)
        all_frames.extend(frames)
        all_durations.extend(durations)

    if not all_frames:
        raise RuntimeError("No frames generated")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=all_frames[1:],
        duration=all_durations,
        loop=0,
        optimize=True,
    )
    all_frames[0].save(OUT_PREVIEW)

    print(f"Created: {OUT_PATH}")
    print(f"Created: {OUT_PREVIEW}")
    print(f"Phases included: {len(PHASES)}")
    print(f"Total frames: {len(all_frames)}")


if __name__ == "__main__":
    main()
