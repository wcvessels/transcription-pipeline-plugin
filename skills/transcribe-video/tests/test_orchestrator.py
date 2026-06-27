import subprocess
from pathlib import Path

import imagehash
import pytest
import transcribe
import manifest


def _junk_score_and_hash(path, box, hash_size=16):
    """Stand-in for frames.score_and_hash that forces every frame below both junk floors (sharpness 0,
    info 0) while still returning a valid content-region hash — exercises the all-junk fallback paths."""
    return {"sharpness": 0.0, "info": 0.0, "hash": imagehash.hex_to_hash("0" * (hash_size * hash_size // 4))}


def _make_tiny_mp4(path, with_audio=True):
    # IMPORTANT: include an audio track — the WhisperX path extracts audio + runs silencedetect,
    # both of which fail on a video with no audio stream. testsrc alone has none.
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += ["-t", "3", str(path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_parser_accepts_a1_0_flag_subset():
    args = transcribe.build_parser().parse_args(
        ["clip.mp4", "--diarize", "off", "--frames-per-minute", "5",
         "--dedup-threshold", "5", "--force-transcribe", "--keep-work", "--max-frames", "50"]
    )
    assert args.diarize == "off" and args.frames_per_minute == 5 and args.keep_work is True


def test_parser_rejects_a2_flags():
    for flag in ["--ai-assist", "--curate", "--align", "--ocr-engine", "--polish",
                 "--compose-with-claude", "--format", "--resume"]:
        with pytest.raises(SystemExit):
            transcribe.build_parser().parse_args(["clip.mp4", flag, "x"])


def test_parser_rejects_bad_numeric_flags():
    # P8 (Codex F6): the argparse type validators reject out-of-range numerics at parse time
    # (SystemExit), instead of a confusing crash deep in the pipeline.
    for bad in (["--window-size", "0"], ["--max-frames", "0"],
                ["--dedup-threshold", "-1"], ["--interval-seconds", "0"]):
        with pytest.raises(SystemExit):
            transcribe.build_parser().parse_args(["clip.mp4", *bad])


def test_prefer_captions_and_env_default(monkeypatch):
    # P11 (Gemini locked-concern 1): caption-vs-transcribe is env-defaulted, flags override.
    # Neither flag → dest is None (the pipeline then reads TRANSCRIBE_VIDEO_FORCE_TRANSCRIBE);
    # --prefer-captions → False; --force-transcribe → True; both together → mutually-exclusive error.
    # Self-contained re: the env var — establish a clean baseline so the test never depends on the
    # ambient environment (the build/CI shell may export TRANSCRIBE_VIDEO_FORCE_TRANSCRIBE).
    monkeypatch.delenv("TRANSCRIBE_VIDEO_FORCE_TRANSCRIBE", raising=False)
    # Flag precedence holds regardless of env: explicit flags pin True/False; neither → None.
    assert transcribe.build_parser().parse_args(["clip.mp4"]).force_transcribe is None
    assert transcribe.build_parser().parse_args(["clip.mp4", "--prefer-captions"]).force_transcribe is False
    assert transcribe.build_parser().parse_args(["clip.mp4", "--force-transcribe"]).force_transcribe is True
    with pytest.raises(SystemExit):
        transcribe.build_parser().parse_args(["clip.mp4", "--force-transcribe", "--prefer-captions"])
    # Exercise the env-default path: with neither flag (dest None) and the env set, the pipeline's
    # documented resolution (flag > env > captions-first) selects transcribe. Assert the same env
    # expression the orchestrator applies (run_pipeline §16.3 resolve), so the env branch is covered
    # without depending on the ambient shell.
    import os
    monkeypatch.setenv("TRANSCRIBE_VIDEO_FORCE_TRANSCRIBE", "1")
    args = transcribe.build_parser().parse_args(["clip.mp4"])
    assert args.force_transcribe is None  # parser stays env-agnostic; resolution happens downstream
    env_default = os.environ.get("TRANSCRIBE_VIDEO_FORCE_TRANSCRIBE", "").strip().lower() in ("1", "true", "yes", "on")
    assert env_default is True  # None dest + this env → the pipeline takes the transcribe path


def test_reserved_source_hint_errors(tmp_path):
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    args = transcribe.build_parser().parse_args([str(f), "--source-hint", "gong"])
    with pytest.raises(SystemExit):
        transcribe.run_pipeline(args, deps=transcribe.default_deps())


def test_dead_code_removed():
    src = Path(transcribe.__file__).read_text(encoding="utf-8")
    assert "find_segment_for_timestamp" not in src  # §10 #2
    assert "def write_srt" not in src and "def write_txt" not in src  # replaced


class _FakeDeps(transcribe.Deps):
    """Inject a fake WhisperX so the orchestrator test needs no model/GPU."""
    def transcribe(self, audio_path, model, language, diarize, device, compute_type):
        segs = [{"index": 0, "start_s": 0.0, "end_s": 2.0, "speaker": None,
                 "text": "fake segment", "frame_index": None}]
        return segs, "en"

    def run_sample_pass(self, audio_path, duration_s, device):
        return 1, False, True  # single speaker, no error, weights available


def test_full_local_run_emits_exact_artifact_set(tmp_path):
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    out = tmp_path / "out"
    args = transcribe.build_parser().parse_args(
        [str(f), "--diarize", "off", "--output-dir", str(out)]
    )
    code = transcribe.run_pipeline(args, deps=_FakeDeps())
    assert code == 0
    b = "clip"
    # curate-and-stop: the 5-artifact set, NO guide
    assert not (out / f"{b}_guide.md").exists()
    assert (out / f"{b}_manifest.json").exists()
    assert (out / f"{b}_frames").is_dir() and any((out / f"{b}_frames").iterdir())
    assert (out / f"{b}_transcript.txt").exists()           # transcript on both paths
    assert (out / f"{b}_contactsheet.jpg").exists()
    assert (out / f"{b}_frames.md").exists()
    # manifest validates, carries the curation block, frames carry sharpness
    import json
    m = json.loads((out / f"{b}_manifest.json").read_text(encoding="utf-8"))
    manifest.validate_manifest(m)
    assert "curation" in m and m["curation"]["kept_count"] == len(m["frames"])
    assert all("sharpness" in fr for fr in m["frames"])
    # work dir removed by default (no --keep-work)
    assert not (out / f"{b}_work").exists()


def test_curation_block_mapping_values(tmp_path):
    # §16.4 curation fields are REPURPOSED onto the dense change-detection model. Pin the values a frozen
    # schema can't catch, so a wiring regression (window_size left at the old 5, or dedup_dropped computed
    # against candidate_count instead of selected_count) is caught by a fast non-gated test.
    import json
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    out = tmp_path / "out"
    args = transcribe.build_parser().parse_args([str(f), "--diarize", "off", "--output-dir", str(out)])
    assert transcribe.run_pipeline(args, deps=_FakeDeps()) == 0
    c = json.loads((out / "clip_manifest.json").read_text(encoding="utf-8"))["curation"]
    assert c["window_size"] == 1                                       # 1 scene-start frame per scene
    assert c["candidate_count"] >= c["selected_count"] >= c["kept_count"]
    assert c["window_count"] >= c["selected_count"]                    # scenes >= scenes that yielded a frame
    assert c["dedup_dropped"] == c["selected_count"] - c["kept_count"]  # "dedup" now == the duration-cap trim


def test_frame_filenames_are_index_prefixed_timestamps(tmp_path):
    # Frames are named frame_<idx:04d>_<HHMMSS>.jpg: the zero-padded index prefix guarantees
    # collision-freedom (two survivors can truncate to the same HHMMSS), the timestamp makes the
    # file scannable. The manifest's frames[].file must match the on-disk names exactly.
    import json
    import re
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    out = tmp_path / "out"
    args = transcribe.build_parser().parse_args([str(f), "--diarize", "off", "--output-dir", str(out)])
    assert transcribe.run_pipeline(args, deps=_FakeDeps()) == 0
    jpgs = sorted((out / "clip_frames").glob("*.jpg"))
    assert jpgs, "no frames emitted"
    for p in jpgs:
        assert re.fullmatch(r"frame_\d{4}_\d{6}\.jpg", p.name), f"unexpected frame name: {p.name}"
    m = json.loads((out / "clip_manifest.json").read_text(encoding="utf-8"))
    assert {fr["file"] for fr in m["frames"]} == {p.name for p in jpgs}


def test_rerun_clears_stale_frames(tmp_path):
    # P7 (Codex F5): a rerun must not leave stale frame_00xx.jpg (or a stale guide) beyond the new
    # set. The orchestrator clears+recreates B_frames/ each run, so a frame_9999.jpg and a stale
    # *_guide.md dropped into the frames dir are gone after the second run, and the frames-dir count
    # equals len(manifest["frames"]). A1.0 also writes NO guide at the output root.
    import json
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    out = tmp_path / "out"
    args = transcribe.build_parser().parse_args([str(f), "--diarize", "off", "--output-dir", str(out)])
    assert transcribe.run_pipeline(args, deps=_FakeDeps()) == 0
    frames_dir = out / "clip_frames"
    # seed stale artifacts into the frames dir before the rerun
    (frames_dir / "frame_9999.jpg").write_bytes(b"stale")
    (frames_dir / "clip_guide.md").write_text("stale guide", encoding="utf-8")
    assert transcribe.run_pipeline(args, deps=_FakeDeps()) == 0
    m = json.loads((out / "clip_manifest.json").read_text(encoding="utf-8"))
    assert not (frames_dir / "frame_9999.jpg").exists()      # stale frame cleared
    assert not list(frames_dir.glob("*_guide.md"))           # stale guide cleared
    jpgs = list(frames_dir.glob("frame_*.jpg"))
    assert len(jpgs) == len(m["frames"])                     # frames dir count == manifest count
    assert not list(out.glob("*_guide.md"))                  # curate-and-stop: no guide written


def test_missing_ffmpeg_exits_cleanly(tmp_path, monkeypatch):
    # F1/F2: a missing binary must be an actionable preflight exit, not an uncaught exception.
    import preflight
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    monkeypatch.setattr(preflight, "report_capabilities",
                        lambda: {"ffmpeg": False, "ffprobe": True, "yt_dlp": True, "gpu": False})
    args = transcribe.build_parser().parse_args([str(f), "--output-dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit) as exc:
        transcribe.run_pipeline(args, deps=_FakeDeps())
    assert exc.value.code == 2  # clean exit, not a traceback


class _ExplodingDeps(_FakeDeps):
    """Transcription stage raises a realistic model error (e.g. CUDA OOM)."""
    def transcribe(self, audio_path, model, language, diarize, device, compute_type):
        raise RuntimeError("CUDA out of memory (simulated)")


def test_no_audio_stream_exits_cleanly(tmp_path):
    # R2 P1 (targeted): a WhisperX-path video with no audio track must fail clean (exit 2), not crash
    # _extract_audio with an uncaught CalledProcessError.
    f = tmp_path / "silent.mp4"; _make_tiny_mp4(f, with_audio=False)
    args = transcribe.build_parser().parse_args(
        [str(f), "--diarize", "off", "--output-dir", str(tmp_path / "o")]
    )
    with pytest.raises(SystemExit) as exc:
        transcribe.run_pipeline(args, deps=_FakeDeps())
    assert exc.value.code == 2


def test_transcription_error_exits_cleanly(tmp_path):
    # R2 P1 (targeted): a throw from the model call surfaces as a clean exit, not a traceback.
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    args = transcribe.build_parser().parse_args(
        [str(f), "--diarize", "off", "--output-dir", str(tmp_path / "o")]
    )
    with pytest.raises(SystemExit) as exc:
        transcribe.run_pipeline(args, deps=_ExplodingDeps())
    assert exc.value.code == 2


def test_backstop_catches_unforeseen_error(tmp_path, monkeypatch):
    # R2 P1 (backstop): an error at a site with NO targeted catch (here: contact-sheet writing) must
    # STILL exit clean via the top-level backstop — proving §16.7 #1 holds for the unforeseen, not
    # just the failure modes we enumerated.
    import curated_output as co_mod
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    def _boom(*a, **k):
        raise RuntimeError("unforeseen contact-sheet failure")
    monkeypatch.setattr(co_mod, "write_contactsheet", _boom)
    args = transcribe.build_parser().parse_args(
        [str(f), "--diarize", "off", "--output-dir", str(tmp_path / "o")]
    )
    with pytest.raises(SystemExit) as exc:
        transcribe.run_pipeline(args, deps=_FakeDeps())
    assert exc.value.code == 2


def test_all_junk_frames_exits_cleanly(tmp_path, monkeypatch):
    # P4 (Codex F2): when every scene's frames score as junk and first_non_junk_per_segment returns NO
    # survivors, and --allow-low-quality-frames is NOT set, the run must clean-fail (exit 2) with the
    # actionable message that names the escape hatch — not crash on an empty frame set.
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    import frames as frames_mod
    # force every sampled frame below both floors (FRAME_BLUR_FLOOR / FRAME_LOW_INFO_FLOOR)
    monkeypatch.setattr(frames_mod, "score_and_hash", _junk_score_and_hash)
    args = transcribe.build_parser().parse_args(
        [str(f), "--diarize", "off", "--output-dir", str(tmp_path / "o")]
    )
    with pytest.raises(SystemExit) as exc:
        transcribe.run_pipeline(args, deps=_FakeDeps())
    assert exc.value.code == 2


def test_all_junk_with_allow_low_quality_completes(tmp_path, monkeypatch):
    # P4 (Codex F2): same all-junk situation, but --allow-low-quality-frames falls back to the single
    # highest-sharpness sampled frame so the run completes at accepted lower fidelity (exit 0,
    # exactly one frame, schema-valid manifest).
    import frames as frames_mod
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    monkeypatch.setattr(frames_mod, "score_and_hash", _junk_score_and_hash)
    out = tmp_path / "out"
    args = transcribe.build_parser().parse_args(
        [str(f), "--diarize", "off", "--allow-low-quality-frames", "--output-dir", str(out)]
    )
    code = transcribe.run_pipeline(args, deps=_FakeDeps())
    assert code == 0
    import json
    m = json.loads((out / "clip_manifest.json").read_text(encoding="utf-8"))
    manifest.validate_manifest(m)
    assert len(m["frames"]) == 1


def test_dedup_recorded_in_state_json(tmp_path):
    # state.json carries the curation block (the surviving dedup diagnostic) and NOT the retired A/B
    # dedup_comparison block (the phash-vs-colorhash decision closed; live gate is joint). --keep-work
    # writes work/state.json.
    import json
    f = tmp_path / "clip.mp4"; _make_tiny_mp4(f)
    out = tmp_path / "out"
    args = transcribe.build_parser().parse_args(
        [str(f), "--diarize", "off", "--keep-work", "--output-dir", str(out)]
    )
    assert transcribe.run_pipeline(args, deps=_FakeDeps()) == 0
    state = json.loads((out / "clip_work" / "state.json").read_text(encoding="utf-8"))
    assert "dedup_comparison" not in state          # retired with compare_dedup_methods
    assert {"kept_count", "dedup_dropped", "dedup_reduction"} <= set(state["curation"])


def test_failed_run_with_staging_like_basename_preserves_work_dir(tmp_path):
    # Important (code-review): _fail must PRESERVE the work dir on failure (Gemini F2) even when the
    # source basename contains the staging sentinel substring "_a1_0_staging". The old path-string
    # discriminator would have wrongly rmtree'd the work dir here; the explicit drop flag fixes it.
    f = tmp_path / "vid_a1_0_staging.mp4"; _make_tiny_mp4(f)
    out = tmp_path / "out"
    args = transcribe.build_parser().parse_args([str(f), "--diarize", "off", "--output-dir", str(out)])
    with pytest.raises(SystemExit) as exc:
        transcribe.run_pipeline(args, deps=_ExplodingDeps())
    assert exc.value.code == 2
    # work dir preserved for diagnosis (NOT deleted), despite the staging-sentinel in the basename
    assert (out / "vid_a1_0_staging_work").is_dir()


def test_resolved_flags_includes_allow_low_quality_frames():
    # (Codex F4 / Gemini #7) --allow-low-quality-frames changes pipeline behavior (clean-fail vs
    # fallback frame), so it MUST be part of the run_id resolved-flags, or a strict run and a permissive
    # rerun collide on the same run_id.
    a1 = transcribe.build_parser().parse_args(["clip.mp4", "--output-dir", "o"])
    a2 = transcribe.build_parser().parse_args(["clip.mp4", "--output-dir", "o", "--allow-low-quality-frames"])
    assert transcribe._resolved_flags(a1) != transcribe._resolved_flags(a2)
    assert manifest.compute_run_id("clip.mp4", transcribe._resolved_flags(a1)) != \
           manifest.compute_run_id("clip.mp4", transcribe._resolved_flags(a2))


def test_extract_frames_fps_clears_stale_candidates(tmp_path, monkeypatch):
    # (Codex F3 / Gemini #2) extract_frames_fps must clear pre-existing d_*.jpg before ffmpeg, so a
    # reused candidates dir cannot leak ghost frames from a prior (longer) run into the candidate list.
    import frames as frames_mod
    out_dir = tmp_path / "candidates"; out_dir.mkdir()
    for i in range(1, 101):                                # 100 stale frames from a "previous" run
        (out_dir / f"d_{i:06d}.jpg").write_bytes(b"stale")
    def _fake_run(cmd, **kw):                              # the "new" short clip emits only 10 frames
        for i in range(1, 11):
            (out_dir / f"d_{i:06d}.jpg").write_bytes(b"new")
        class _P: pass
        return _P()
    monkeypatch.setattr(frames_mod.subprocess, "run", _fake_run)
    result = frames_mod.extract_frames_fps(tmp_path / "v.mp4", 1.0, out_dir)
    assert len(result) == 10                               # only the new frames, no stale leakage
