"""Curated-output writers (§16.2, curate-and-stop): B_transcript.txt (both paths) + B_frames.md
index. Pure Python/deterministic — the local inputs the prosumer compose tier consumes. No guide
composition here (that migrated up to the prosumer tier)."""
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
        f"{len(frames_sorted)} frames (one per on-screen scene, dense change-detection).", "",
        "| # | Timestamp | Sharpness | Scene cut | Frame |",
        "|---|---|---|---|---|",
    ]
    for fr in frames_sorted:
        ts = fmt_ts(fr["timestamp_s"])
        cut = "yes" if fr.get("is_scene_cut") else ""
        link = f"![{ts}]({frames_dir_rel}/{fr['file']})"
        lines.append(f"| {fr['index']} | {ts} | {fr['sharpness']:.1f} | {cut} | {link} |")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
