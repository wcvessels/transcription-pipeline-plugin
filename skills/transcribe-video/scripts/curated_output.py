"""Curated-output writers (§16.2, curate-and-stop): B_transcript.txt (both paths) + B_frames.md
index + B_contactsheet.jpg. Pure Python/deterministic — the local inputs the prosumer compose
tier consumes. No guide composition here (that migrated up to the prosumer tier)."""
from pathlib import Path

from timefmt import fmt_ts


def write_transcript(segments, output_path):
    """Verbatim transcript, speaker-grouped when speakers exist. Written on BOTH paths (decision #3):
    captions segments have speaker=None, so output degrades to plain paragraphs with no labels."""
    lines = []
    current = "\x00"  # sentinel no real speaker label equals
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = seg.get("speaker")
        if spk is None:
            lines.append(text)
        else:
            if spk != current:
                if lines:
                    lines.append("")          # blank line between speaker turns
                lines.append(f"[{spk}]")
                current = spk
            lines.append(text)
    Path(output_path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_frames_index(frame_records, frames_dir_rel, output_path):
    """B_frames.md: a flat index of the curated frames — one row per kept frame with timestamp,
    sharpness, scene-cut flag, and image link. Deterministic, no prose."""
    frames_sorted = sorted(frame_records, key=lambda f: f["timestamp_s"])
    lines = [
        "# Curated frames", "",
        f"{len(frames_sorted)} frames (perceptual-hash deduped, best-of-window).", "",
        "| # | Timestamp | Sharpness | Scene cut | Frame |",
        "|---|---|---|---|---|",
    ]
    for fr in frames_sorted:
        ts = fmt_ts(fr["timestamp_s"])
        cut = "yes" if fr.get("is_scene_cut") else ""
        link = f"![{ts}]({frames_dir_rel}/{fr['file']})"
        lines.append(f"| {fr['index']} | {ts} | {fr['sharpness']:.1f} | {cut} | {link} |")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contactsheet(frame_records, frames_dir, output_path, cols=4, thumb_w=320, pad=8, label_h=18):
    """B_contactsheet.jpg: a thumbnail grid of the curated frames, each captioned with its timestamp.
    Pure Pillow (default bitmap font — no font files needed); deterministic for a given frame set."""
    from PIL import Image, ImageDraw
    frames_sorted = sorted(frame_records, key=lambda f: f["timestamp_s"])
    if not frames_sorted:
        raise ValueError("no frames to contact-sheet")
    first = Image.open(Path(frames_dir) / frames_sorted[0]["file"])
    thumb_h = max(1, int(thumb_w * first.height / first.width))
    cell_w, cell_h = thumb_w + pad, thumb_h + label_h + pad
    rows = (len(frames_sorted) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, fr in enumerate(frames_sorted):
        r, c = divmod(i, cols)
        x, y = pad + c * cell_w, pad + r * cell_h
        thumb = Image.open(Path(frames_dir) / fr["file"]).convert("RGB").resize((thumb_w, thumb_h))
        sheet.paste(thumb, (x, y))
        draw.text((x + 2, y + thumb_h + 2), fmt_ts(fr["timestamp_s"]), fill=(20, 20, 20))
    sheet.save(output_path, "JPEG", quality=85)
