#!/usr/bin/env python
"""transcribe-video A1.0 — local video file / public URL → CURATED ARTIFACT SET
(frames + transcript + manifest + frames index).

Curate-and-stop vertical slice per DESIGN_video-ingest-plugin.md §16. One linear invocation, no
packaging, no Claude, no handoff state machine, **no guide composition** (that migrated up to the
prosumer tier). Diarization is token-free (the promoted clone). CLI entry point kept stable so
transcribe-video.bat keeps working."""
import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import alignment as alignment_mod
import captions as captions_mod
import curated_output as co_mod
import frames as frames_mod
import manifest as manifest_mod
import preflight as preflight_mod
import resolver as resolver_mod
import timefmt as timefmt_mod
import transcription as tx_mod

TOOL_VERSION = "a1.0"
RESERVED_HINTS = {"m365", "box", "gong", "fireflies"}
# dense change-detection curation knobs (tunable module defaults). §4.4.
FRAME_BLUR_FLOOR = 20.0          # Laplacian-variance floor; below this a frame is "too blurry"
FRAME_LOW_INFO_FLOOR = 2.0       # histogram-entropy floor; below this a frame is "near-blank"
FRAME_SAMPLE_FPS = 1.0           # dense sample rate (frames/sec); --interval-seconds overrides as 1/N
FRAME_HASH_SIZE = 16             # phash side (256-bit); 64-bit (size 8) can't separate held-jitter from a
                                 # real screen change on shared-chrome app screens — the over-collapse fix
FRAME_CONTENT_VAR_FLOOR = 12     # per-pixel range above which a pixel is "content" (vs constant letterbox)
FRAME_CAP_PER_MIN = 40           # runaway backstop: max kept frames per MINUTE of video (scales the
                                 # default --max-frames by duration); only bites pathological full-motion


def _positive_int(raw):
    v = int(raw)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def _nonnegative_int(raw):
    v = int(raw)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def _positive_float(raw):
    v = float(raw)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


class Deps:
    """Seam for the heavy stages so the orchestrator is testable without a GPU."""
    def transcribe(self, audio_path, model, language, diarize, device, compute_type):
        return tx_mod.transcribe_segments(audio_path, model, language, diarize, device, compute_type)

    def run_sample_pass(self, audio_path, duration_s, device):
        # token-free: no HF token. Returns (distinct_speakers, errored, weights_available).
        return tx_mod.run_sample_pass(audio_path, duration_s, device)


def default_deps():
    return Deps()


def build_parser():
    ap = argparse.ArgumentParser(description="Video → curated artifact set (A1.0 vertical slice).")
    ap.add_argument("source", help="Local video path or one public video URL")
    # kept from v1
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language")
    ap.add_argument("--keep-audio", action="store_true")
    ap.add_argument("--output-dir")
    ap.add_argument("--scene-threshold", type=float, default=0.3)
    ap.add_argument("--max-frames", type=_positive_int, default=None,
                    help="absolute cap on kept frames; default scales with duration "
                         f"({FRAME_CAP_PER_MIN}/min runaway backstop), so long sessions are not capped")
    ap.add_argument("--interval-seconds", type=_positive_float,
                    help="dense sample period in seconds (overrides the 1 fps default as rate = 1/N)")
    # A1 new, in A1.0
    ap.add_argument("--dedup-threshold", type=_nonnegative_int, default=3,
                    help="scene-change threshold: a new scene starts when the frame's phash@16 (256-bit) "
                         "changes by more than this Hamming distance; lower = more captures; 0 keeps every frame")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--force-transcribe", dest="force_transcribe", action="store_true", default=None,
                     help="force WhisperX even if captions exist (overrides the env default)")
    grp.add_argument("--prefer-captions", dest="force_transcribe", action="store_false", default=None,
                     help="force the captions path when captions exist (overrides the env default)")
    ap.add_argument("--keep-work", action="store_true", help="retain B_work/ scratch dir")
    ap.add_argument("--diarize", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--source-hint", choices=["url", "file", *sorted(RESERVED_HINTS)])
    return ap


def _resolved_flags(args):
    # run_id inputs (§4.9): only the flags that change the curated output. Cosmetic/--keep-* flags are
    # excluded so they never perturb the run_id (resumability key).
    return {"diarize": args.diarize, "max_frames": args.max_frames,
            "scene_threshold": args.scene_threshold, "dedup_threshold": args.dedup_threshold,
            "force_transcribe": args.force_transcribe,
            "interval_seconds": args.interval_seconds, "model": args.model}


def _fail(message: str, code: int = 2, cleanup=None, drop: bool = False) -> "SystemExit":
    """Print an actionable error and exit cleanly (§16.7 #1). On failure we PRESERVE the work dir for
    diagnosis (Gemini F2) unless the caller explicitly asks to drop it (drop=True), which is used only
    for the pre-resolve download staging scratch. The caller states intent — we never infer a
    destructive rmtree from the path string."""
    print(f"Error: {message}", file=sys.stderr)
    if cleanup is not None:
        if drop:
            shutil.rmtree(cleanup, ignore_errors=True)   # pre-resolve download scratch: safe to drop
        else:
            print(f"Work dir kept for inspection: {cleanup}", file=sys.stderr)
    raise SystemExit(code)


def run_pipeline(args, deps) -> int:
    """Top-level entry. Guarantees §16.7 #1 (no uncaught exception escapes): every failure becomes
    an actionable message + non-zero SystemExit via _fail(), never a raw traceback. The targeted
    catches inside handle the expected real-world failures; this backstop covers the unforeseen."""
    try:
        return _run_pipeline_inner(args, deps)
    except SystemExit:
        raise  # already a clean, intentional exit from _fail()
    except Exception as e:  # backstop (round-2 P1): nothing unforeseen reaches the user as a traceback
        raise _fail(f"unexpected failure ({type(e).__name__}): {e}")


def _run_pipeline_inner(args, deps) -> int:
    proc_t0 = time.monotonic()
    # 1. classify FIRST so out_dir is chosen before any download (avoids a cross-drive rename
    #    and the chicken-and-egg of needing the basename to name the work dir).
    try:
        kind = resolver_mod.classify_source(args.source, args.source_hint)
    except (resolver_mod.ReservedFeatureError, resolver_mod.UnsupportedSourceError) as e:
        raise _fail(str(e))

    # Report capabilities once, then gate the *binaries needed to even resolve the source*
    # BEFORE any download (F1: a missing yt-dlp must be an actionable preflight error, not a
    # raw subprocess failure). PreflightError is caught here, not left uncaught (F2).
    caps = preflight_mod.report_capabilities()
    try:
        preflight_mod.require_for_resolution(kind, caps)
    except preflight_mod.PreflightError as e:
        raise _fail(str(e))

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    elif kind == "file":
        out_dir = Path(args.source).resolve().parent
    else:
        out_dir = Path.cwd().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage the download INSIDE out_dir (same drive → safe rename) then resolve.
    staging = out_dir / "_a1_0_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        local_path, meta = resolver_mod.resolve(args.source, staging, source_hint=args.source_hint)
    except (resolver_mod.ReservedFeatureError, resolver_mod.UnsupportedSourceError) as e:
        raise _fail(str(e), cleanup=staging, drop=True)
    except subprocess.CalledProcessError as e:
        # F1's sibling (R2 P1): yt-dlp/ffprobe ran but exited non-zero (bad URL, network, private/geo).
        raise _fail(f"source download/probe failed (yt-dlp/ffprobe exit {e.returncode}). "
                    "Check the URL, your network, and that the video is public.", cleanup=staging, drop=True)

    basename = meta["basename"]
    run_id = manifest_mod.compute_run_id(args.source, _resolved_flags(args))
    work = out_dir / f"{basename}_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    staging.rename(work)
    # After the rename, downloaded artifacts moved with the dir — repoint paths for the URL case.
    if kind != "file":
        local_path = work / local_path.name
        if meta.get("caption_path"):
            meta["caption_path"] = str(work / Path(meta["caption_path"]).name)

    frames_dir = out_dir / f"{basename}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)   # rerun: no stale frames beyond the new set (§16.2 exact)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 2. decide caption-vs-transcribe, then gate the mode-specific requirements. This MUST be
    #    post-decision: a captioned URL under --diarize on but without --force-transcribe takes the
    #    captions path. Resolve the env default when NEITHER flag was given (Gemini locked-concern 1):
    #    precedence is flag > env > captions-first. --force-transcribe/--prefer-captions set
    #    args.force_transcribe True/False; None means "fall back to the env default".
    import os
    if args.force_transcribe is None:
        args.force_transcribe = os.environ.get("TRANSCRIBE_VIDEO_FORCE_TRANSCRIBE", "").strip().lower() in ("1", "true", "yes", "on")
    use_captions = bool(meta.get("caption_path")) and not args.force_transcribe
    mode = "captions" if use_captions else ("diarize_on" if args.diarize == "on"
           else "diarize_off" if args.diarize == "off" else "diarize_auto")
    try:
        preflight_mod.require_for_mode(mode, caps)
    except preflight_mod.PreflightError as e:
        raise _fail(str(e), cleanup=(None if args.keep_work else work))

    # 3. frames: dense change-detection curation (§4.4). ONE ffmpeg pass samples at FRAME_SAMPLE_FPS;
    #    trim any constant letterbox borders; score + junk-filter each frame; segment the timeline into
    #    held SCENES by the phash delta between consecutive frames (a screen change); keep the FIRST
    #    non-junk frame of each scene (scene-START, the alignment key) → cap by duration. ffmpeg scene
    #    cuts are computed only as alignment anchors (decoupled from sampling, decision #3).
    scene_times = frames_mod.detect_scenes(local_path, args.scene_threshold)   # alignment anchors only
    sample_fps = (1.0 / args.interval_seconds) if args.interval_seconds else FRAME_SAMPLE_FPS
    cut_tol = 0.5 / sample_fps   # a frame is "at" a cut if one falls within half a sample period of it
    cand_dir = work / "candidates"
    dense = frames_mod.extract_frames_fps(local_path, sample_fps, cand_dir)   # [(path, ts)], one decode
    candidate_count = len(dense)
    if not dense:
        raise _fail("no frames could be extracted from the source at all (decode failed). "
                    "The file may be corrupt.", cleanup=(None if args.keep_work else work))
    # content box trims constant letterbox/pillarbox borders (a no-op on full-bleed screen recordings,
    # where nothing is globally constant; helps genuinely letterboxed sources). Change is measured on
    # the box; on full-bleed clips that is the whole frame and the threshold carries completeness.
    # computed over ALL dense frames (incrementally, no stacking) so a localized change in any frame
    # cannot be cropped out before hashing — a 24-frame subsample created exactly that under-capture blind spot.
    box = frames_mod.content_box_from_paths([p for p, _ in dense], var_threshold=FRAME_CONTENT_VAR_FLOOR)
    # score every dense frame + compute its change hash (on the box)
    records = []
    for p, ts in dense:
        s = frames_mod.score_and_hash(p, box, hash_size=FRAME_HASH_SIZE)
        records.append({"file": str(p), "timestamp_s": ts, "sharpness": s["sharpness"],
                        "info": s["info"], "hash": s["hash"]})
    # segment into held scenes by frame-to-frame change; keep the first non-junk frame per scene (scene-start)
    segments = frames_mod.segment_scenes([r["hash"] for r in records], args.dedup_threshold)
    survivors = frames_mod.first_non_junk_per_segment(records, segments, FRAME_BLUR_FLOOR, FRAME_LOW_INFO_FLOOR)
    # coverage invariant: first_non_junk_per_segment yields exactly one frame per detected scene (all-junk
    # scenes fall back to their least-blurry frame), and >=1 scene exists because `dense` was non-empty above.
    assert survivors, "coverage invariant violated: a detected scene yielded no frame"
    selected_count = len(survivors)
    # over-capture cap (§4.4): default scales with duration (FRAME_CAP_PER_MIN/min) so long sessions are
    # never capped by a flat number; --max-frames N overrides with an absolute ceiling. A2 culls the rest.
    # un-probeable containers report duration_s == 0.0; the dense frame span (count/fps) is a reliable
    # proxy, so the cap never collapses a real clip to one frame on a missing container duration.
    cap_duration = meta["duration_s"] or (len(dense) / sample_fps)
    kept = frames_mod.decimate(
        survivors, frames_mod.frame_cap(cap_duration, args.max_frames, FRAME_CAP_PER_MIN))
    # copy kept frames into B_frames/ as frame_0001_HHMMSS.jpg … and build FrameRecords. Field-whitelist:
    # only §16.4 schema fields are written; the in-memory change `hash` is stringified into `phash`.
    frame_records = []
    for i, rec in enumerate(kept):
        ts = rec["timestamp_s"]
        # index prefix = collision guard (two frames can truncate to the same HHMMSS); the HHMMSS suffix
        # makes the frame scannable on disk. fmt_clock truncates, win32-safe (no ':').
        dest = frames_dir / f"frame_{i + 1:04d}_{timefmt_mod.fmt_clock(ts)}.jpg"
        shutil.copy(rec["file"], dest)
        frame_records.append({
            "index": i, "timestamp_s": ts, "file": dest.name,
            "is_scene_cut": frames_mod.frame_at_cut(ts, scene_times, cut_tol),   # nearest-cut alignment anchor
            "phash": str(rec["hash"]), "sharpness": round(float(rec["sharpness"]), 3),
        })
    kept_count = len(frame_records)
    # curation block maps the dense pipeline onto the locked §16.4 fields: window == scene (window_count =
    # scenes detected, window_size = 1 frame/scene); dedup_dropped/dedup_reduction now report the
    # decimation (over-capture cap) trim — no absolute-similarity dedup stage exists anymore.
    dedup_dropped = selected_count - kept_count
    curation = {
        "candidate_count": candidate_count, "window_count": len(segments),
        "window_size": 1, "selected_count": selected_count,
        "dedup_dropped": dedup_dropped, "kept_count": kept_count,
        "dedup_reduction": round(dedup_dropped / selected_count, 4) if selected_count else 0.0,
    }

    # 4. transcription branch
    silence_gaps, caption_cues, speaker_turns_list = [], [], []
    diar_reason = "captions_no_audio"
    diarization = "off"
    model_used = None
    language = "en"
    speaker_count = 0

    if use_captions:
        cues = captions_mod.parse_vtt(Path(meta["caption_path"]).read_text(encoding="utf-8"))
        segments = captions_mod.to_segments(cues)
        caption_cues = captions_mod.cue_boundaries(cues)
        path_kind = "captions"
    else:
        path_kind = "whisperx"
        device = "cuda" if caps.get("gpu") else "cpu"
        compute_type = tx_mod.auto_compute_type(tx_mod.detect_gpu_name())
        audio_path = work / "audio.wav"
        try:
            _extract_audio(local_path, audio_path)
        except subprocess.CalledProcessError:
            # R2 P1: a video with no audio track makes ffmpeg exit non-zero here; fail clean.
            raise _fail("audio extraction failed. The video may have no audio track, which the "
                        "WhisperX path requires. Use a captioned source or check the file.",
                        cleanup=(None if args.keep_work else work))
        # F3: audio is present on this path → compute the silence-gap anchor signal (§4.5 #3).
        silence_gaps = tx_mod.detect_silence_gaps(audio_path)
        # resolve diarization. Token-free: --diarize auto ALWAYS runs the sample pass (no HF
        # precondition, no preflight degrade). The pass returns (speakers, errored, weights_available);
        # a cold-offline weights-unavailable run degrades to OFF here (auto_degraded_weights_unavailable)
        # rather than failsafe-ON into a diarization that also can't load.
        if args.diarize == "auto":
            speakers, errored, weights_ok = deps.run_sample_pass(audio_path, meta["duration_s"], device)
            if not weights_ok:
                diarize_bool, diar_reason = False, "auto_degraded_weights_unavailable"
            else:
                diarize_bool, diar_reason = tx_mod.decide_diarization("auto", speakers, errored)
        else:
            diarize_bool, diar_reason = tx_mod.decide_diarization(args.diarize, 0, False)
        try:
            segments, language = deps.transcribe(audio_path, args.model, args.language,
                                                 diarize_bool, device, compute_type)
        except Exception as e:
            # R2 P1: model load / CUDA OOM / align failure surfaces as a clean exit, not a traceback.
            raise _fail(f"transcription failed ({type(e).__name__}): {e}",
                        cleanup=(None if args.keep_work else work))
        diarization = "on" if diarize_bool else "off"
        model_used = args.model
        speaker_turns_list = tx_mod.speaker_turns(segments)
        speaker_count = len({s.get("speaker") for s in segments if s.get("speaker")})

    # 5. alignment (joint-signal, local) — STAYS; writes segments[].frame_index, the compose bridge
    anchors = alignment_mod.build_anchors(
        scene_cuts=scene_times, speaker_turns=speaker_turns_list,
        silence_gaps=silence_gaps, caption_cues=caption_cues, duration=meta["duration_s"])
    segments = alignment_mod.align(frame_records, segments, anchors, meta["duration_s"])
    anchor_counts = {"scene_cuts": len(scene_times), "speaker_turns": len(speaker_turns_list),
                     "silence_gaps": len(silence_gaps), "caption_cues": len(caption_cues)}

    # 6. write the curated artifact set (4 artifacts, all always produced): transcript + frames.md index
    #    + manifest. frames/ was populated in step 3. NO contactsheet (A1.2 Shape-1: it crashed past
    #    ~1272 frames on PIL's 65500px ceiling and was a useless mosaic past ~300; frames/ + frames.md
    #    fully cover visual review). frames.md is a normal write — a failure is a clean exit via the
    #    top-level backstop, not a silent degrade. NO guide (that migrated to the compose tier).
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    transcript_path = out_dir / f"{basename}_transcript.txt"
    co_mod.write_transcript(segments, transcript_path)
    frames_index_path = out_dir / f"{basename}_frames.md"
    co_mod.write_frames_index(frame_records, frames_dir.name, frames_index_path)

    processing_s = time.monotonic() - proc_t0 - float(meta.get("download_s", 0.0))
    manifest_obj = manifest_mod.build_manifest(
        source={"uri": meta["uri"], "type": meta["type"], "title": meta.get("title"),
                "duration_s": meta["duration_s"], "width": meta["width"], "height": meta["height"],
                "fps": meta["fps"], "codec": meta["codec"]},
        run={"run_id": run_id,
             "generated_at": generated_at, "tool_version": TOOL_VERSION,
             "host_os": manifest_mod.detect_host_os(),
             "download_s": float(meta.get("download_s", 0.0)), "processing_s": max(0.0, processing_s)},
        transcription={"path": path_kind, "model": model_used, "diarization": diarization,
                       "diarization_reason": diar_reason, "language": language,
                       "speaker_count": speaker_count},
        frames=frame_records, curation=curation, segments=segments, anchor_counts=anchor_counts,
        artifacts={"manifest_json": f"{basename}_manifest.json", "frames_dir": frames_dir.name,
                   "transcript_txt": transcript_path.name,
                   "frames_index_md": frames_index_path.name},
    )
    manifest_path = out_dir / f"{basename}_manifest.json"
    import json
    manifest_path.write_text(json.dumps(manifest_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    # state.json (§16.5): trivial single-run record. The dedup/curation counts live in the manifest
    # `curation` block (dedup_reduction is recorded for diagnostics — no hard floor as of the round-9
    # recalibration; dedup correctness is gated synthetically), so this is just a resumability stub.
    (work / "state.json").write_text(
        json.dumps({"run_id": run_id, "path": path_kind, "curation": curation}, indent=2),
        encoding="utf-8")

    # 7. cleanup. §16.5: audio.wav is kept with --keep-audio OR --keep-work; the whole scratch
    # dir (candidates + audio) is kept only with --keep-work.
    if args.keep_audio and not args.keep_work:
        audio_src = work / "audio.wav"          # exists only on the WhisperX path
        if audio_src.exists():
            shutil.copy(audio_src, out_dir / f"{basename}.wav")
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[frames] dense change-detection (phash@{FRAME_HASH_SIZE} delta>{args.dedup_threshold}): "
          f"{candidate_count} sampled → {curation['window_count']} scenes → {selected_count} "
          f"scene-start frames → kept {kept_count} after the duration cap "
          f"(trimmed {dedup_dropped}).", file=sys.stderr)
    print(f"Curated artifacts written to: {out_dir}")
    return 0


def _extract_audio(video_path, audio_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    # The .bat shim invokes this under cmd, where Python's console stdout/stderr default to the
    # legacy cp1252 codec on Windows — which crashes on the non-ASCII glyphs in --help text and in
    # actionable error messages (→, —, §). Force UTF-8 on the real console streams at process entry
    # so the entry point never dies with a UnicodeEncodeError instead of printing usage/errors.
    # (Tests call build_parser/run_pipeline directly under pytest's own UTF-8 capture, so this only
    # affects the live CLI path.)
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass  # already non-reconfigurable (e.g. piped) — leave as-is
    args = build_parser().parse_args()
    raise SystemExit(run_pipeline(args, default_deps()))


if __name__ == "__main__":
    main()
