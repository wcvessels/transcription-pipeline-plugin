import pytest
import manifest


def test_load_schema_has_expected_top_level_required():
    schema = manifest.load_schema()
    assert schema["required"] == [
        "schema_version", "source", "run", "transcription", "frames", "curation",
        "segments", "alignment", "artifacts"
    ]


def test_valid_captions_manifest_passes(valid_captions_manifest):
    manifest.validate_manifest(valid_captions_manifest)  # must not raise


def test_valid_whisperx_manifest_passes(valid_whisperx_manifest):
    manifest.validate_manifest(valid_whisperx_manifest)  # must not raise


def test_missing_required_key_is_rejected(valid_captions_manifest):
    del valid_captions_manifest["alignment"]
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_unknown_extra_key_is_rejected(valid_captions_manifest):
    valid_captions_manifest["surprise"] = 1
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_bad_source_type_enum_is_rejected(valid_captions_manifest):
    valid_captions_manifest["source"]["type"] = "ftp"
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_bad_diarization_reason_enum_is_rejected(valid_captions_manifest):
    valid_captions_manifest["transcription"]["diarization_reason"] = "because_i_said_so"
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_transcript_txt_required_string_on_captions_path(valid_captions_manifest):
    # redesign decision #3: both paths emit B_transcript.txt; null is no longer allowed anywhere
    valid_captions_manifest["artifacts"]["transcript_txt"] = None
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_empty_frames_array_is_rejected(valid_captions_manifest):
    valid_captions_manifest["frames"] = []  # schema requires minItems: 1
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_captions_path_with_model_is_rejected(valid_captions_manifest):
    # §16.4 invariant (F8): captions path must have model == null
    valid_captions_manifest["transcription"]["model"] = "large-v3"
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_legacy_guide_md_artifact_is_rejected(valid_captions_manifest):
    # curate-and-stop: guide_md is gone from the artifact set; additionalProperties:false rejects it
    valid_captions_manifest["artifacts"]["guide_md"] = "demo_guide.md"
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_transcript_txt_required_string_on_whisperx_path(valid_whisperx_manifest):
    valid_whisperx_manifest["artifacts"]["transcript_txt"] = None
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_whisperx_manifest)


def test_missing_curation_block_is_rejected(valid_captions_manifest):
    del valid_captions_manifest["curation"]
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_dedup_reduction_above_one_is_rejected(valid_captions_manifest):
    valid_captions_manifest["curation"]["dedup_reduction"] = 1.5  # schema maximum is 1
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_frame_missing_sharpness_is_rejected(valid_captions_manifest):
    del valid_captions_manifest["frames"][0]["sharpness"]
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_whisperx_path_with_null_model_is_rejected(valid_whisperx_manifest):
    valid_whisperx_manifest["transcription"]["model"] = None
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_whisperx_manifest)


def test_bad_generated_at_format_is_rejected(valid_captions_manifest):
    # round-2 P2: format: date-time must actually be enforced (format_checker + rfc3339-validator),
    # else an invalid generated_at silently passes the §16.7 #3 schema gate.
    valid_captions_manifest["run"]["generated_at"] = "not-a-timestamp"
    with pytest.raises(manifest.ManifestValidationError):
        manifest.validate_manifest(valid_captions_manifest)


def test_shipped_schema_matches_canonical_hash():
    # in-house guard: the shipped schema file must stay byte-identical (normalized: LF, no trailing
    # newline) to DESIGN §16.4 / PLAN Task 1. Locks the single-source invariant as an automated test.
    import hashlib
    from pathlib import Path
    raw = Path(manifest.SCHEMA_PATH).read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert h == "9fbc7e26b0414b031042cbe2f979cc13fb7896ed084f46c06912bd91391832c9", (
        f"manifest-1.0.schema.json drifted from the canonical §16.4 hash: {h}")
