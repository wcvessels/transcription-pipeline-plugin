"""Preflight: capability report + per-mode requirement check. §16.6 / §4.8.
Reports everything; blocks only on what the chosen mode needs. Neither the WhisperX model nor the
token-free diarization weights are ever a hard block (both auto-fetch on first use). Token-free
diarization means NO mode requires HF_TOKEN or a pyannote license — those capabilities are gone."""
import shutil


class PreflightError(Exception):
    pass


def report_capabilities() -> dict:
    """JSON-serializable capability report; never blocks. Reports only what the per-mode gates
    actually consume — the WhisperX model and diarization weights auto-fetch on first use, and that
    first-run download cost is surfaced just-in-time at model load (not as a report-only key here)."""
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,  # resolver.probe_metadata shells out to it
        # runtime invokes `sys.executable -m yt_dlp` (resolver._download_url), so the importable
        # MODULE is what matters — a PATH-only yt-dlp binary would pass preflight then crash (R3 P1).
        "yt_dlp": _module_present("yt_dlp"),
        "gpu": _gpu_present(),
    }


def require_for_resolution(kind: str, caps: dict) -> None:
    """Gate the binaries needed to even fetch the source, BEFORE any download (F1). ffmpeg/ffprobe
    always; yt-dlp for the URL branch. Raises PreflightError with an actionable message instead of
    letting a missing binary become a raw subprocess crash."""
    if not caps.get("ffmpeg"):
        raise PreflightError("ffmpeg is required and not on PATH. Install ffmpeg and re-run.")
    if not caps.get("ffprobe"):
        raise PreflightError("ffprobe is required (source metadata probe) and not on PATH. "
                             "It ships with ffmpeg; install the full ffmpeg build and re-run.")
    if kind == "yt-dlp" and not caps.get("yt_dlp"):
        raise PreflightError("yt-dlp is required for URL ingestion and not importable by this "
                             "interpreter. Install it here (python -m pip install yt-dlp) and re-run.")


def require_for_mode(mode: str, caps: dict) -> None:
    """Block only on hard requirements for `mode` (§16.6 table). Token-free: every transcription/
    diarization mode needs only ffmpeg (+ffprobe) — the WhisperX model and the diarization weights
    auto-fetch and are never a hard block. Captions additionally needs yt-dlp (URL ingestion)."""
    if not caps.get("ffmpeg"):
        raise PreflightError("ffmpeg is required and not on PATH. Install ffmpeg and re-run.")
    if not caps.get("ffprobe"):
        raise PreflightError("ffprobe is required (ships with ffmpeg) and not on PATH. "
                             "Install the full ffmpeg build and re-run.")
    if mode == "captions":
        if not caps.get("yt_dlp"):
            raise PreflightError("yt-dlp is required for URL ingestion and not importable "
                                 "(python -m pip install yt-dlp).")
        return
    if mode in ("diarize_off", "diarize_on", "diarize_auto"):
        return  # ffmpeg only; WhisperX model + diarization weights auto-fetch, never a hard block
    raise PreflightError(f"Unknown mode: {mode}")


# ---- capability probes ----

def _module_present(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _gpu_present():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
