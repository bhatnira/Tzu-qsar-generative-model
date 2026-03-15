#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "outputs/generated/animations/final_structures/minimal_diffs"
OUT_PATH = IN_DIR / "minimal_combined_summary.gif"

SOURCES = [
    ("Scaffold Preservation", IN_DIR / "scaffold_preservation_minimal.gif"),
    ("Scaffold Non-Preservation", IN_DIR / "scaffold_non_preservation_minimal.gif"),
    ("Bioisostere Replacement", IN_DIR / "bioisostere_replacement_minimal.gif"),
]


def first_frame(path: Path) -> Image.Image:
    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            return frame.convert("RGB")
    raise RuntimeError(f"No frames found in {path}")


def fit_with_white_bg(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", target_size, "white")
    fitted = img.copy()
    fitted.thumbnail(target_size, Image.Resampling.LANCZOS)
    x = (target_size[0] - fitted.width) // 2
    y = (target_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def main() -> None:
    for _, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(f"Missing source GIF: {path}")

    panels = []
    for title, path in SOURCES:
        frame = first_frame(path)
        panels.append((title, fit_with_white_bg(frame, (500, 420))))

    canvas_size = (1600, 620)
    panel_positions = [(30, 120), (550, 120), (1070, 120)]

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    frames = []
    durations = []

    for highlight in range(3):
        canvas = Image.new("RGB", canvas_size, "white")
        draw = ImageDraw.Draw(canvas)

        draw.text((30, 24), "Minimal Comparison Summary: Plain vs RL vs CL", fill="black", font=font)

        for i, ((title, panel_img), (x, y)) in enumerate(zip(panels, panel_positions)):
            canvas.paste(panel_img, (x, y))
            draw.text((x + 8, y - 26), title, fill="black", font=font)
            border_color = "black" if i == highlight else (190, 190, 190)
            border_width = 4 if i == highlight else 1
            for w in range(border_width):
                draw.rectangle([x - w, y - w, x + 500 + w, y + 420 + w], outline=border_color)

        frames.append(canvas)
        durations.append(1500)

    # Hold final frame a bit longer
    frames.append(frames[-1].copy())
    durations.append(2000)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    print(f"Created: {OUT_PATH}")


if __name__ == "__main__":
    main()
