import manifest


def _frames():
    return [{"index": 0, "timestamp_s": 1.0, "file": "frame_0001.jpg", "is_scene_cut": True,
             "phash": "ab", "sharpness": 88.0}]


def _segments():
    return [{"index": 0, "start_s": 0.0, "end_s": 3.0, "speaker": None, "text": "hi", "frame_index": 0}]


def _curation():
    return {"candidate_count": 5, "window_count": 1, "window_size": 5, "selected_count": 1,
            "dedup_dropped": 0, "kept_count": 1, "dedup_reduction": 0.0}


def _artifacts(prefix):
    return {"manifest_json": f"{prefix}_manifest.json", "frames_dir": f"{prefix}_frames",
            "transcript_txt": f"{prefix}_transcript.txt", "frames_index_md": f"{prefix}_frames.md"}


def test_run_id_is_deterministic_for_same_inputs():
    a = manifest.compute_run_id("clip.mp4", {"diarize": "auto", "max_frames": 100})
    b = manifest.compute_run_id("clip.mp4", {"max_frames": 100, "diarize": "auto"})  # key order irrelevant
    assert a == b and len(a) >= 8


def test_build_captions_manifest_validates():
    m = manifest.build_manifest(
        source={"uri": "https://youtube.com/x", "type": "yt-dlp", "title": "T",
                "duration_s": 180.0, "width": 1280, "height": 720, "fps": 30.0, "codec": "h264"},
        run={"run_id": "abc12345", "generated_at": "2026-06-08T12:00:00Z", "tool_version": "a1.0",
             "host_os": "windows", "download_s": 10.0, "processing_s": 20.0},
        transcription={"path": "captions", "model": None, "diarization": "off",
                       "diarization_reason": "captions_no_audio", "language": "en", "speaker_count": 0},
        frames=_frames(), curation=_curation(), segments=_segments(),
        anchor_counts={"scene_cuts": 1, "speaker_turns": 0, "silence_gaps": 0, "caption_cues": 2},
        artifacts=_artifacts("x"),
    )
    assert m["run"]["wall_clock_s"] == 30.0  # download + processing
    assert m["curation"]["window_size"] == 5
    manifest.validate_manifest(m)  # must not raise (transcript_txt present on captions path now)


def test_build_whisperx_manifest_single_speaker_validates():
    m = manifest.build_manifest(
        source={"uri": "clip.mp4", "type": "file", "title": "clip",
                "duration_s": 600.0, "width": 1920, "height": 1080, "fps": 30.0, "codec": "h264"},
        run={"run_id": "def67890", "generated_at": "2026-06-08T12:00:00Z", "tool_version": "a1.0",
             "host_os": "windows", "download_s": 0.0, "processing_s": 240.0},
        transcription={"path": "whisperx", "model": "large-v3", "diarization": "off",
                       "diarization_reason": "auto_single_speaker", "language": "en", "speaker_count": 1},
        frames=_frames(), curation=_curation(), segments=_segments(),
        anchor_counts={"scene_cuts": 4, "speaker_turns": 0, "silence_gaps": 3, "caption_cues": 0},
        artifacts=_artifacts("clip"),
    )
    manifest.validate_manifest(m)
    assert m["transcription"]["diarization_reason"] == "auto_single_speaker"


def test_detect_host_os_is_enum_value():
    assert manifest.detect_host_os() in {"windows", "macos", "linux"}


def test_build_rejects_captions_with_model():
    # F8 (surviving invariant): build_manifest enforces captions→model null before writing
    import pytest
    with pytest.raises(manifest.ManifestValidationError):
        manifest.build_manifest(
            source={"uri": "u", "type": "yt-dlp", "title": None, "duration_s": 10.0,
                    "width": 1, "height": 1, "fps": 1.0, "codec": "h264"},
            run={"run_id": "x", "generated_at": "2026-06-08T12:00:00Z", "tool_version": "a1.0",
                 "host_os": "windows", "download_s": 1.0, "processing_s": 1.0},
            transcription={"path": "captions", "model": "large-v3", "diarization": "off",
                           "diarization_reason": "captions_no_audio", "language": "en", "speaker_count": 0},
            frames=_frames(), curation=_curation(), segments=_segments(),
            anchor_counts={"scene_cuts": 0, "speaker_turns": 0, "silence_gaps": 0, "caption_cues": 1},
            artifacts=_artifacts("g"))


def test_build_rejects_whisperx_with_null_model():
    # F8 (surviving invariant): whisperx→model string. (The old path↔transcript_raw invariant is
    # gone — transcript_txt is now an unconditional required string on both paths.)
    import pytest
    with pytest.raises(manifest.ManifestValidationError):
        manifest.build_manifest(
            source={"uri": "clip.mp4", "type": "file", "title": "clip", "duration_s": 10.0,
                    "width": 1, "height": 1, "fps": 1.0, "codec": "h264"},
            run={"run_id": "x", "generated_at": "2026-06-08T12:00:00Z", "tool_version": "a1.0",
                 "host_os": "windows", "download_s": 0.0, "processing_s": 1.0},
            transcription={"path": "whisperx", "model": None, "diarization": "off",
                           "diarization_reason": "forced_off", "language": "en", "speaker_count": 0},
            frames=_frames(), curation=_curation(), segments=_segments(),
            anchor_counts={"scene_cuts": 0, "speaker_turns": 0, "silence_gaps": 0, "caption_cues": 0},
            artifacts=_artifacts("g"))
