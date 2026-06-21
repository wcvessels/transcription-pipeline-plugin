"""A1.0 source resolver: Tier-1 only (local file + public yt-dlp URL). §4.1 / §16.3."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}  # == §16.1 + SUPPORTED_SOURCES_MSG + v1 auto-discovery
URL_RE = re.compile(r"^https?://", re.I)

# Full reserved --source-hint enum (§4.1). Only url/file are functional in A1.0.
FUNCTIONAL_HINTS = {"url", "file"}
RESERVED_CONNECTOR_HINTS = {"m365", "box", "gong", "fireflies"}
ALL_HINTS = FUNCTIONAL_HINTS | RESERVED_CONNECTOR_HINTS

SUPPORTED_SOURCES_MSG = (
    "Unsupported source. A1.0 supports exactly two inputs:\n"
    "  - a local file path (.mp4, .mov, .mkv, .webm, .avi)\n"
    "  - one public video URL (YouTube and other yt-dlp sites)\n"
    "Cloud connectors (M365/Box/Gong/Fireflies) are an A1.x feature, not available yet."
)


class UnsupportedSourceError(Exception):
    pass


class ReservedFeatureError(Exception):
    pass


def _is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def _is_video_file(p: Path) -> bool:
    """R3 P3: A1.0 accepts only the §16.1 video extensions, not any existing file."""
    return p.suffix.lower() in VIDEO_EXTS


def classify_source(source: str, source_hint=None) -> str:
    """Return 'file' or 'yt-dlp'. Raise on reserved hints or unrecognized input."""
    if source_hint is not None and source_hint in RESERVED_CONNECTOR_HINTS:
        raise ReservedFeatureError(
            f"--source-hint '{source_hint}' is an A1.x feature, not available yet."
        )
    if source_hint == "file":
        return "file"
    if source_hint == "url":
        return "yt-dlp"
    if source_hint is not None and source_hint not in ALL_HINTS:
        raise UnsupportedSourceError(f"Unknown --source-hint '{source_hint}'.")

    # No (or non-decisive) hint: classify by the source string itself.
    if _is_url(source):
        return "yt-dlp"
    p = Path(source)
    if p.exists() and p.is_file():
        if not _is_video_file(p):
            raise UnsupportedSourceError(
                f"'{p.name}' is not a supported video file. A1.0 accepts: "
                + ", ".join(sorted(VIDEO_EXTS)) + ".")
        return "file"
    raise UnsupportedSourceError(SUPPORTED_SOURCES_MSG)


def probe_metadata(video_path: Path) -> dict:
    """ffprobe → duration/width/height/fps/codec."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,codec_name,avg_frame_rate",
         "-show_entries", "format=duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(proc.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    num, _, den = (stream.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) != 0 else 0.0
    return {
        "duration_s": float(fmt.get("duration", 0.0) or 0.0),
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "fps": fps,
        "codec": stream.get("codec_name", "") or "",
    }


def _download_url(url: str, workdir: Path) -> tuple:
    """yt-dlp download (Brad's flag set, §4.1). Returns (local_path, caption_path|None, info|{})."""
    out_template = str(workdir / "source.%(ext)s")
    subprocess.run(
        [sys.executable, "-m", "yt_dlp",
         "-N", "8",
         "-f", "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
         "--merge-output-format", "mp4",
         "--write-info-json", "--write-subs", "--write-auto-subs",
         "--sub-langs", "en,en-US,en-GB,en-orig", "--sub-format", "vtt", "--convert-subs", "vtt",
         "--no-playlist", "-o", out_template, url],
        check=True,
    )
    vids = [p for p in workdir.glob("source.*") if p.suffix.lower() in VIDEO_EXTS]
    if not vids:
        raise UnsupportedSourceError("yt-dlp completed but produced no video file.")
    vtts = sorted(workdir.glob("source*.vtt"))
    info = workdir / "source.info.json"
    info_data = json.loads(info.read_text(encoding="utf-8")) if info.exists() else {}
    return vids[0], (vtts[0] if vtts else None), info_data


def resolve(source: str, workdir: Path, source_hint=None) -> tuple:
    """Return (local_video_path, metadata_dict). metadata carries everything downstream needs."""
    kind = classify_source(source, source_hint)
    if kind == "file":
        p = Path(source).resolve()
        if not p.exists():
            raise UnsupportedSourceError(f"Video file not found: {p}")
        if not _is_video_file(p):
            raise UnsupportedSourceError(
                f"'{p.name}' is not a supported video file. A1.0 accepts: "
                + ", ".join(sorted(VIDEO_EXTS)) + ".")
        md = probe_metadata(p)
        md.update({"type": "file", "uri": str(p), "title": p.stem,
                   "caption_path": None, "download_s": 0.0, "basename": p.stem})
        return p, md

    # yt-dlp URL path
    t0 = time.monotonic()
    local_path, caption_path, info = _download_url(source, workdir)
    download_s = time.monotonic() - t0
    md = probe_metadata(local_path)
    md.update({
        "type": "yt-dlp", "uri": source,
        "title": info.get("title"),
        "caption_path": str(caption_path) if caption_path else None,
        "download_s": download_s,
        "basename": _safe_basename(info.get("id") or local_path.stem),
    })
    return local_path, md


def _safe_basename(raw: str) -> str:
    """Lowercase, keep [a-z0-9_-]; used as artifact prefix B."""
    return re.sub(r"[^a-z0-9_-]+", "", str(raw).lower()) or "video"
