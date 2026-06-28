import sys
import copy
from pathlib import Path

# Put scripts/ on sys.path so tests can `import manifest`, `import alignment`, etc.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pytest


def _valid_captions_manifest():
    """Minimal manifest for the captions path (null model; transcript present on BOTH paths now)."""
    return {
        "schema_version": "1.1",
        "source": {
            "uri": "https://youtube.com/watch?v=abc", "type": "yt-dlp", "title": "Demo",
            "duration_s": 180.0, "width": 1280, "height": 720, "fps": 30.0, "codec": "h264",
        },
        "run": {
            "run_id": "abc123def456", "generated_at": "2026-06-08T12:00:00Z",
            "tool_version": "a1.0", "host_os": "windows",
            "wall_clock_s": 40.0, "download_s": 12.0, "processing_s": 28.0,
        },
        "transcription": {
            "path": "captions", "model": None, "diarization": "off",
            "diarization_reason": "captions_no_audio", "language": "en", "speaker_count": 0,
        },
        "frames": [
            {"index": 0, "timestamp_s": 1.0, "file": "frame_0001.jpg", "is_scene_cut": True,
             "phash": "ff00ff00ff00ff00", "sharpness": 142.5},
        ],
        "curation": {
            "candidate_count": 30, "window_count": 6, "window_size": 5, "selected_count": 6,
            "dedup_dropped": 5, "kept_count": 1, "dedup_reduction": 0.8333,
        },
        "segments": [
            {"index": 0, "start_s": 0.0, "end_s": 3.0, "speaker": None, "text": "hello", "frame_index": 0},
        ],
        "alignment": {
            "mode": "joint",
            "anchor_counts": {"scene_cuts": 1, "speaker_turns": 0, "silence_gaps": 0, "caption_cues": 2},
        },
        "artifacts": {
            "manifest_json": "demo_manifest.json", "frames_dir": "demo_frames",
            "transcript_txt": "demo_transcript.txt", "frames_index_md": "demo_frames.md",
        },
    }


@pytest.fixture
def valid_captions_manifest():
    return copy.deepcopy(_valid_captions_manifest())


@pytest.fixture
def valid_whisperx_manifest():
    """Same shape but the WhisperX path: real model, auto_single_speaker. transcript_txt is
    already present in the base fixture (both paths emit it now), so only transcription changes."""
    m = _valid_captions_manifest()
    m["transcription"] = {
        "path": "whisperx", "model": "large-v3", "diarization": "off",
        "diarization_reason": "auto_single_speaker", "language": "en", "speaker_count": 1,
    }
    return copy.deepcopy(m)
