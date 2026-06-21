import pytest
import preflight


def _caps(**over):
    # token-free diarization: no hf_token / pyannote_license capabilities anymore
    base = {"ffmpeg": True, "ffprobe": True, "yt_dlp": True, "gpu": True,
            "model_cached": True, "diarization_weights": True}
    base.update(over)
    return base


def test_captions_only_needs_ffmpeg_and_ytdlp():
    preflight.require_for_mode("captions", _caps(model_cached=False, gpu=False,
                                                 diarization_weights=False))  # must not raise


def test_captions_only_missing_ffmpeg_blocks():
    with pytest.raises(preflight.PreflightError):
        preflight.require_for_mode("captions", _caps(ffmpeg=False))


def test_diarize_off_needs_only_ffmpeg():
    preflight.require_for_mode("diarize_off", _caps(model_cached=False,
                                                    diarization_weights=False))  # must not raise


def test_diarize_on_needs_only_ffmpeg_no_hf():
    # token-free (redesign): forced diarization no longer requires HF_TOKEN or a pyannote license
    preflight.require_for_mode("diarize_on", _caps(diarization_weights=False))  # must not raise


def test_diarize_auto_never_blocks():
    preflight.require_for_mode("diarize_auto", _caps(diarization_weights=False))  # must not raise


def test_resolution_url_needs_ytdlp_before_download():
    # F1: a missing yt-dlp must surface as a preflight error at resolution time, not a subprocess crash
    with pytest.raises(preflight.PreflightError) as exc:
        preflight.require_for_resolution("yt-dlp", _caps(yt_dlp=False))
    assert "yt-dlp" in str(exc.value).lower()


def test_resolution_file_needs_only_ffmpeg():
    preflight.require_for_resolution("file", _caps(yt_dlp=False, model_cached=False,
                                                   diarization_weights=False))


def test_resolution_missing_ffmpeg_blocks_any_source():
    with pytest.raises(preflight.PreflightError):
        preflight.require_for_resolution("file", _caps(ffmpeg=False))


def test_resolution_missing_ffprobe_blocks():
    # R3 P1: probe_metadata() shells out to ffprobe at resolution time, so a missing ffprobe must
    # block here, not crash mid-resolve. (ffmpeg present does NOT imply ffprobe present.)
    with pytest.raises(preflight.PreflightError):
        preflight.require_for_resolution("file", _caps(ffprobe=False))


def test_report_yt_dlp_reflects_module_not_path(monkeypatch):
    # R3 P1: the downloader runs `sys.executable -m yt_dlp`, so a PATH-only yt-dlp binary with no
    # importable module must report yt_dlp=False — else preflight passes then the download crashes.
    monkeypatch.setattr(preflight.shutil, "which",
                        lambda name: "/usr/bin/yt-dlp" if name == "yt-dlp" else None)
    monkeypatch.setattr(preflight, "_module_present", lambda name: False)
    assert preflight.report_capabilities()["yt_dlp"] is False


def test_report_capabilities_returns_all_keys():
    rep = preflight.report_capabilities()
    for k in ["ffmpeg", "ffprobe", "yt_dlp", "gpu", "model_cached", "diarization_weights"]:
        assert k in rep
    # token-free: HF capabilities are gone from the report
    assert "hf_token" not in rep and "pyannote_license" not in rep


def test_report_json_is_serializable():
    import json
    json.dumps(preflight.report_capabilities())  # must not raise
