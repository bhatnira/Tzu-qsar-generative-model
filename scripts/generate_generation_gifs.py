from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ANIM_DIR = ROOT / "outputs" / "generated" / "animations"
TARGETS = {
    "mol2mol": ROOT / "outputs" / "generated" / "mol2mol",
    "linkinvent": ROOT / "outputs" / "generated" / "linkinvent",
    "linkinvent_transformer_pubchem": ROOT / "outputs" / "generated" / "linkinvent_transformer_pubchem",
    "libinvent": ROOT / "outputs" / "generated" / "libinvent",
    "libinvent_transformer_pubchem": ROOT / "outputs" / "generated" / "libinvent_transformer_pubchem",
}

WIDTH = 1100
HEIGHT = 760
BG = (11, 15, 25)
PANEL = (18, 24, 38)
TEXT = (236, 240, 245)
MUTED = (148, 163, 184)
ACCENT = (99, 102, 241)
SUCCESS = (34, 197, 94)
NEW = (250, 204, 21)
GRID = (42, 51, 68)


@dataclass
class Event:
    timestamp: datetime
    relpath: str


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_SUB = load_font(20)
FONT_BODY = load_font(18)
FONT_BODY_BOLD = load_font(18, bold=True)
FONT_SMALL = load_font(15)


def collect_events(folder: Path) -> list[Event]:
    events: list[Event] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            events.append(
                Event(
                    timestamp=datetime.fromtimestamp(path.stat().st_mtime),
                    relpath=path.relative_to(folder).as_posix(),
                )
            )
    events.sort(key=lambda e: (e.timestamp, e.relpath))
    return events


def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font, fill, max_width: int, line_spacing: int = 4) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


def make_frame(name: str, folder: Path, events: list[Event], upto: int) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Background panels
    draw.rounded_rectangle((28, 24, WIDTH - 28, 130), radius=22, fill=PANEL)
    draw.rounded_rectangle((28, 152, WIDTH - 28, HEIGHT - 28), radius=22, fill=PANEL)

    total = len(events)
    shown = upto + 1
    current = events[upto]
    start_time = events[0].timestamp
    end_time = events[-1].timestamp

    draw.text((52, 44), f"{name} generation timeline", font=FONT_TITLE, fill=TEXT)
    draw.text((54, 92), f"Folder: {folder.relative_to(ROOT).as_posix()}", font=FONT_SUB, fill=MUTED)

    # Summary badges
    badges = [
        (f"Files: {total}", ACCENT),
        (f"Frame: {shown}/{total}", SUCCESS),
        (f"Time: {current.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", NEW),
    ]
    bx = WIDTH - 320
    by = 42
    for label, color in badges:
        w = int(draw.textlength(label, font=FONT_SMALL)) + 26
        draw.rounded_rectangle((bx, by, bx + w, by + 28), radius=14, fill=color)
        draw.text((bx + 13, by + 6), label, font=FONT_SMALL, fill=(10, 10, 10))
        by += 36

    # Progress bar
    bar_x1, bar_y1, bar_x2, bar_y2 = 54, 168, WIDTH - 54, 190
    draw.rounded_rectangle((bar_x1, bar_y1, bar_x2, bar_y2), radius=11, fill=GRID)
    progress = shown / total if total else 0
    fill_x = bar_x1 + int((bar_x2 - bar_x1) * progress)
    draw.rounded_rectangle((bar_x1, bar_y1, fill_x, bar_y2), radius=11, fill=ACCENT)
    draw.text((54, 202), f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}   End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", font=FONT_SMALL, fill=MUTED)

    # Current file panel
    draw.rounded_rectangle((54, 236, WIDTH - 54, 322), radius=18, fill=(28, 36, 54))
    draw.text((74, 254), "Newest event", font=FONT_BODY_BOLD, fill=NEW)
    draw.text((74, 282), current.relpath, font=FONT_BODY, fill=TEXT)

    # File list
    draw.text((54, 348), "Cumulative files", font=FONT_BODY_BOLD, fill=TEXT)
    y = 380
    max_lines = 13
    start_index = max(0, shown - max_lines)
    visible = list(enumerate(events[start_index:shown], start=start_index))
    for idx, event in visible:
        color = NEW if idx == upto else TEXT
        prefix = "+" if idx == upto else "•"
        line = f"{prefix} {event.timestamp.strftime('%H:%M:%S')}  {event.relpath}"
        draw.text((70, y), line, font=FONT_BODY, fill=color)
        y += 26

    if start_index > 0:
        draw.text((70, y + 6), f"… {start_index} earlier files omitted", font=FONT_SMALL, fill=MUTED)

    # Footer
    footer = "Animated from file modification times; each frame adds the next generated artifact."
    draw.text((54, HEIGHT - 60), footer, font=FONT_SMALL, fill=MUTED)
    return img


def save_gif(name: str, folder: Path, events: list[Event]) -> Path:
    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    frames = [make_frame(name, folder, events, i) for i in range(len(events))]
    if not frames:
        raise ValueError(f"No files found under {folder}")

    out_path = ANIM_DIR / f"{name}_generation.gif"
    durations = [700] * max(1, len(frames) - 1) + [1400]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    return out_path


def main(names: Iterable[str] | None = None) -> None:
    selected = list(names) if names else list(TARGETS.keys())
    generated: list[Path] = []
    for name in selected:
        folder = TARGETS[name]
        events = collect_events(folder)
        out = save_gif(name, folder, events)
        generated.append(out)
        print(out.relative_to(ROOT).as_posix())

    print(f"Generated {len(generated)} GIFs in {ANIM_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
