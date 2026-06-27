"""Manifest loading, validation, and (Task 11) assembly against manifest-1.0.schema.json."""
import json
from functools import lru_cache
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "manifest-1.0.schema.json"

# `format: date-time` (run.generated_at) is only ENFORCED when (a) a format checker is attached
# AND (b) rfc3339-validator is installed. jsonschema otherwise treats `format` as a no-op
# annotation, so an invalid timestamp would silently pass the §16.7 #3 schema gate (round-2 P2).
# rfc3339-validator is pinned in requirements-a1.0.txt for exactly this reason.
_FORMAT_CHECKER = jsonschema.Draft202012Validator.FORMAT_CHECKER


class ManifestValidationError(Exception):
    """Raised when a manifest object does not conform to the §16.4 schema."""


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_manifest(obj: dict) -> None:
    """Validate obj against the manifest schema. Raises ManifestValidationError on failure."""
    validator = jsonschema.Draft202012Validator(load_schema(), format_checker=_FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        lines = [f"  - {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]
        raise ManifestValidationError("Manifest failed schema validation:\n" + "\n".join(lines))


import hashlib
import platform


def detect_host_os() -> str:
    sysname = platform.system().lower()
    if sysname.startswith("win"):
        return "windows"
    if sysname == "darwin":
        return "macos"
    return "linux"


def compute_run_id(source: str, resolved_flags: dict) -> str:
    """Stable hash of (source + resolved flags); order-independent, delimiter-unambiguous. §4.9."""
    canon = json.dumps([source, {k: resolved_flags[k] for k in sorted(resolved_flags)}], sort_keys=True, default=str)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]


def _check_invariants(obj: dict) -> None:
    """Producer-side cross-field invariants (F8, surviving subset). The schema's allOf also enforces
    these; failing here gives a clearer error at construction. transcript_txt is now an unconditional
    required string (both paths), so there is no longer a path↔transcript invariant to check."""
    t = obj["transcription"]
    if t["path"] == "captions":
        if t["model"] is not None:
            raise ManifestValidationError("captions path must have transcription.model == null")
        if t["diarization"] != "off" or t["diarization_reason"] != "captions_no_audio":
            raise ManifestValidationError(
                "captions path requires diarization 'off' and reason 'captions_no_audio'")
    elif t["path"] == "whisperx":
        if not isinstance(t["model"], str):
            raise ManifestValidationError("whisperx path must have a transcription.model string")


def build_manifest(source, run, transcription, frames, curation, segments, anchor_counts, artifacts) -> dict:
    """Assemble + validate a manifest object. `run` carries download_s/processing_s; wall_clock_s is
    computed here so the two timing fields are single-sourced (§16.4). `curation` is the dense
    change-detection summary block; dedup_reduction is recorded for diagnostics — no hard floor (round-9
    recalibration); change-detection correctness is gated by the segment_scenes unit tests (test_frames.py)."""
    run = dict(run)
    run["wall_clock_s"] = float(run["download_s"]) + float(run["processing_s"])
    obj = {
        "schema_version": "1.0",
        "source": source,
        "run": run,
        "transcription": transcription,
        "frames": frames,
        "curation": curation,
        "segments": segments,
        "alignment": {"mode": "joint", "anchor_counts": anchor_counts},
        "artifacts": artifacts,
    }
    _check_invariants(obj)   # clear producer-side error before schema validation
    validate_manifest(obj)
    return obj
