from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "outputs" / "generated" / "animations" / "structures"
OUT_DIR = ROOT / "outputs" / "generated" / "animations" / "final_structures"

TARGETS = [
    "mol2mol_structures.gif",
    "linkinvent_structures.gif",
    "linkinvent_transformer_pubchem_structures.gif",
    "libinvent_structures.gif",
    "libinvent_transformer_pubchem_structures.gif",
]


def font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


FNT = font(22, bold=True)


def add_label(frame: Image.Image, text: str) -> Image.Image:
    img = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    pad = 14
    x1, y1 = 18, 18
    w = int(draw.textlength(text, font=FNT)) + pad * 2
    h = 42
    draw.rounded_rectangle((x1, y1, x1 + w, y1 + h), radius=14, fill=(14, 22, 40))
    draw.text((x1 + pad, y1 + 9), text, font=FNT, fill=(230, 236, 245))
    return img


def pick_indices(n: int):
    if n <= 3:
        return list(range(n))
    i1 = max(0, n // 3 - 1)
    i2 = max(i1 + 1, (2 * n) // 3 - 1)
    i3 = n - 1
    return [i1, i2, i3]


def process_gif(src: Path, out: Path):
    with Image.open(src) as im:
        n = getattr(im, "n_frames", 1)
        idxs = pick_indices(n)
        stage_labels = ["Stage end: plain", "Stage end: reinforcement learning", "Stage end: curriculum learning"]

        frames = []
        for j, idx in enumerate(idxs):
            im.seek(idx)
            lbl = stage_labels[min(j, len(stage_labels) - 1)]
            frames.append(add_label(im.copy(), lbl))

        out.parent.mkdir(parents=True, exist_ok=True)
        durations = [1300, 1300, 1800][: len(frames)]
        if len(durations) < len(frames):
            durations += [1400] * (len(frames) - len(durations))
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)


def main():
    generated = []
    for name in TARGETS:
        src = SRC_DIR / name
        if not src.exists():
            print(f"SKIP missing {src.relative_to(ROOT)}")
            continue
        out_name = name.replace("_structures.gif", "_final_structure_path.gif")
        out = OUT_DIR / out_name
        process_gif(src, out)
        generated.append(out)
        print(out.relative_to(ROOT).as_posix())
    print(f"Generated {len(generated)} focused final-path GIFs in {OUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
