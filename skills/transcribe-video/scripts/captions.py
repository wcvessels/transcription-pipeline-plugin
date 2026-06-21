"""VTT caption parsing + verbatim caption capture.

The VTT parser lineage is adapted from Brad Bonanno's `claude-video`
(MIT License, https://github.com/bradautomates/claude-video) — see §4.3.
"""
import re

_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")
_CUE_TIME = re.compile(r"-->")


def _parse_ts(token: str) -> float:
    m = _TS.search(token)
    if not m:
        raise ValueError(f"Bad VTT timestamp: {token!r}")
    h, mn, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mn * 60 + s + ms / 1000.0


def parse_vtt(text: str) -> list:
    """Minimal WebVTT parser → [{start_s, end_s, text}]. Ignores styling/positioning."""
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines or lines[0].strip().upper().startswith("WEBVTT"):
            continue
        time_line = next((ln for ln in lines if _CUE_TIME.search(ln)), None)
        if not time_line:
            continue
        start_tok, _, end_tok = time_line.partition("-->")
        text_lines = lines[lines.index(time_line) + 1:]
        body = " ".join(t.strip() for t in text_lines).strip()
        body = re.sub(r"<[^>]+>", "", body)  # strip inline tags like <c> / <00:00:01.000>
        if not body:
            continue
        cues.append({"start_s": _parse_ts(start_tok), "end_s": _parse_ts(end_tok), "text": body})
    return cues


def to_segments(cues: list) -> list:
    """Index raw cues VERBATIM into SegmentRecord shape — NO content dedup (A1.0 captures faithfully;
    rolling-caption de-overlap is an A2 compose-tier responsibility). frame_index filled by alignment."""
    segs = []
    for i, c in enumerate(cues):
        segs.append({"index": i, "start_s": float(c["start_s"]), "end_s": float(c["end_s"]),
                     "speaker": None, "text": c["text"].strip(), "frame_index": None})
    return segs


def cue_boundaries(cues: list) -> list:
    """Sorted unique cue start/end times — coarse anchor signal on the captions path (§4.5)."""
    bounds = set()
    for c in cues:
        bounds.add(round(float(c["start_s"]), 3))
        bounds.add(round(float(c["end_s"]), 3))
    return sorted(bounds)
