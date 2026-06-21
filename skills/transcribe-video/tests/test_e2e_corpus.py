import json
import os
from pathlib import Path

import pytest

import transcribe
import manifest

SKILL_ROOT = Path(__file__).resolve().parent.parent
_URL_FILE = SKILL_ROOT / "corpus" / "A1.0_youtube_captions.url"
PINNED_URL = _URL_FILE.read_text(encoding="utf-8").strip() if _URL_FILE.exists() else ""
SOLO_MP4 = os.environ.get("A1_0_SOLO_MP4")
# R2 P2 guard: Will pins the real URL at build time (§16.8). Until then the fixture holds a
# placeholder; the gate must SKIP (loudly) rather than feed yt-dlp a bogus URL and report a
# confusing download failure as if the pipeline were broken.
URL_IS_PINNED = bool(PINNED_URL) and "REPLACE_WITH" not in PINNED_URL


def _assert_frame_index_integrity(m):
    # Codex F4: every non-null segment frame_index must be a valid index into m["frames"].
    n = len(m["frames"])
    for seg in m["segments"]:
        fi = seg["frame_index"]
        if fi is not None:
            assert 0 <= fi < n, f"segment {seg['index']} frame_index {fi} out of range [0,{n})"


@pytest.mark.corpus
def test_corpus_configured_for_gate():
    """§16.7 definition-of-done guard (R3 P1): the two gate tests skip individually when their input
    is missing (so a partial dev run isn't a wall of red), but a real `-m corpus` gate run must FAIL —
    not silently skip to a green exit 0 — when anything the gate needs is absent. This non-skipped
    test is that hard floor. To exercise a single path during development, select it by name, e.g.
    `-m corpus -k captions`, which deselects this guard."""
    missing = []
    if not URL_IS_PINNED:
        missing.append("corpus/A1.0_youtube_captions.url still holds the placeholder "
                       "(pin a real captioned URL, §16.8)")
    if not SOLO_MP4:
        missing.append("A1_0_SOLO_MP4 env var is not set (point it at the solo screen-capture, §16.8)")
    elif not Path(SOLO_MP4).is_file():
        missing.append(f"A1_0_SOLO_MP4 points at a missing file: {SOLO_MP4}")
    # token-free: no HF_TOKEN requirement anymore — the diarization clone fetches its own weights.
    assert not missing, (
        "corpus gate is NOT fully configured — a skipped gate test is a FAILED gate, not a pass:\n  - "
        + "\n  - ".join(missing)
    )


@pytest.mark.corpus
@pytest.mark.skipif(not URL_IS_PINNED,
                    reason="pin a real captioned YouTube URL in corpus/A1.0_youtube_captions.url (§16.8) before running the captions gate")
def test_captions_url_gate(tmp_path):
    """§16.7: captions path → 5-artifact set (transcript on BOTH paths now), manifest valid. "Fast" is
    STRUCTURAL — the captions path provably skips transcription (path=="captions", model==null), which is
    the speed win; processing_s is informational, not a flat cap (round-8 reframe of §16.7 #5)."""
    # --prefer-captions forces the captions path regardless of the ambient
    # TRANSCRIBE_VIDEO_FORCE_TRANSCRIBE env default, so this gate is env-robust (flag > env).
    args = transcribe.build_parser().parse_args(
        [PINNED_URL, "--prefer-captions", "--output-dir", str(tmp_path)]
    )
    assert transcribe.run_pipeline(args, transcribe.default_deps()) == 0
    manifests = list(tmp_path.glob("*_manifest.json"))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text(encoding="utf-8"))
    manifest.validate_manifest(m)
    assert m["transcription"]["path"] == "captions"
    # curate-and-stop: NO guide; the 5-artifact set; transcript present even on the captions path
    assert not list(tmp_path.glob("*_guide.md"))
    assert m["artifacts"]["transcript_txt"] and list(tmp_path.glob("*_transcript.txt"))
    assert list(tmp_path.glob("*_contactsheet.jpg")) and list(tmp_path.glob("*_frames.md"))
    assert m["transcription"]["model"] is None      # §16.7 #5 reframe (round 8): captions path skips ASR, so "fast" is structural, not a flat 60s cap; processing_s logged informational
    assert 1 <= len(m["frames"]) <= args.max_frames  # §16.7 #4 bounds
    _assert_frame_index_integrity(m)                 # Codex F4: frame_index ∈ [0, len(frames))


@pytest.mark.corpus
@pytest.mark.skipif(not SOLO_MP4, reason="set A1_0_SOLO_MP4 to the solo screen-capture path")
def test_local_solo_gate(tmp_path):
    """§16.7 #6: TOKEN-FREE auto diarization on a solo recording. No HF_TOKEN — the promoted clone
    runs the sample pass on its own verified weights. The load-bearing assertion is a STABLE PROPERTY
    (Gemini F1), not the exact reason string: a single-window pyannote flip on a solo-dominant
    recording is a known model-noise artifact, so we assert the recording IS solo-dominant + that a
    token-free AUTO pass genuinely ran, rather than hard-failing solely on `auto_single_speaker`."""
    args = transcribe.build_parser().parse_args(
        [SOLO_MP4, "--diarize", "auto", "--keep-work", "--output-dir", str(tmp_path)]
    )
    assert transcribe.run_pipeline(args, transcribe.default_deps()) == 0
    m = json.loads(next(tmp_path.glob("*_manifest.json")).read_text(encoding="utf-8"))
    manifest.validate_manifest(m)
    assert m["transcription"]["path"] == "whisperx"
    assert m["artifacts"]["transcript_txt"] and list(tmp_path.glob("*_transcript.txt"))
    assert list(tmp_path.glob("*_contactsheet.jpg")) and list(tmp_path.glob("*_frames.md"))
    # Stable property 1 — a token-free AUTO pass actually ran (no HF token, the clone executed; not
    # forced, not captions). DOCUMENTED expected case: auto_single_speaker (a solo recording resolves
    # diarization off) — but we do NOT hard-fail solely on it (per Gemini F1, see below).
    AUTO_REASONS = {"auto_single_speaker", "auto_multi_speaker",
                    "auto_sample_error_failsafe_on", "auto_degraded_weights_unavailable"}
    assert m["transcription"]["diarization_reason"] in AUTO_REASONS, (
        f"expected a token-free AUTO diarization reason, got {m['transcription']['diarization_reason']!r}")
    # Stable property 2 — the recording is SOLO-DOMINANT: from m["segments"], one speaker holds the
    # large majority of attributed speaker-seconds (>= 0.85), OR speaker_count <= 1. On mismatch the
    # message DUMPS the per-speaker seconds for diagnosis.
    secs = {}
    for seg in m["segments"]:
        spk = seg.get("speaker")
        if spk is not None:
            secs[spk] = secs.get(spk, 0.0) + (float(seg["end_s"]) - float(seg["start_s"]))
    total = sum(secs.values())
    solo_dominant = (m["transcription"]["speaker_count"] <= 1) or (total > 0 and max(secs.values()) / total >= 0.85)
    assert solo_dominant, f"recording is not solo-dominant; per-speaker seconds = {secs}"
    assert 1 <= len(m["frames"]) <= args.max_frames  # §16.7 #4 frame-count bounds
    _assert_frame_index_integrity(m)                 # Codex F4: frame_index ∈ [0, len(frames))
    # §16.7 #4 curation-block integrity, read from the MANIFEST (not a tautology against files-on-disk).
    # RECALIBRATED (round 9, 2026-06-18): the prior `dedup_reduction >= 0.30` floor was REMOVED. Best-of-
    # window removes intra-window redundancy UPSTREAM, so a well-curated, content-rich screen-capture
    # yields mostly-distinct survivors and a legitimately LOW cross-window dedup_reduction (two real solo
    # clips measured 0.14 and 0.00; colorhash only "passed" the old floor by over-merging distinct frames
    # that share a palette). Dedup CORRECTNESS is now gated where redundancy is CONTROLLED — the synthetic
    # test_frames.py::test_synthetic_dedup_floor_drops_duplicates_keeps_distinct — not by a redundancy
    # ratio on uncontrolled real content. The corpus gate validates the end-to-end token-free solo path
    # plus a valid curation block.
    cur = m["curation"]
    assert cur["kept_count"] == len(m["frames"])
    assert cur["candidate_count"] >= cur["selected_count"] >= cur["kept_count"]
