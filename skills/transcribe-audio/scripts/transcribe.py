#!/usr/bin/env python
"""
WhisperX wrapper: transcribe audio with speaker diarization.

By default writes txt (speaker-labeled), srt (subtitles), and json (full data)
next to the input audio. Use --format to write only one. Uses CUDA when
available, falls back to CPU. Diarization runs fully locally -- models auto-download on first use (no HF account or token).
"""
import argparse
import json as json_mod
import sys
from pathlib import Path


def format_timestamp(seconds, always_include_hours=True):
    assert seconds >= 0, "non-negative timestamp expected"
    ms = round(seconds * 1000.0)
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if always_include_hours else f"{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            speaker = seg.get("speaker", "")
            text = seg.get("text", "").strip()
            label = f"[{speaker}] {text}" if speaker else text
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{label}\n\n")


def write_txt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            if speaker != current_speaker:
                if current_speaker is not None:
                    f.write("\n")
                f.write(f"[{speaker}] ")
                current_speaker = speaker
            f.write(text + " ")
        f.write("\n")


def symlink_privilege_hint(exc, model_name):
    """Actionable message for Windows symlink-privilege failures (WinError 1314)
    while downloading a not-yet-cached ASR model; None for any other error.

    huggingface_hub caches model snapshots as symlinks into a blob store.
    Without Developer Mode (or admin) Windows denies symlink creation, and
    hub 0.36.2's threaded snapshot downloads can crash on the denial instead
    of falling back to file copies. Already-cached models never hit this.
    """
    if getattr(exc, "winerror", None) != 1314:
        return None
    return (
        f"Error: downloading model '{model_name}' failed because Windows denied "
        "symlink creation in the huggingface cache (WinError 1314).\n"
        "This is a download race: a plain re-run usually completes (finished files are kept).\n"
        "Permanent fix (either works):\n"
        "  - Enable Windows Developer Mode (Settings > Update & Security > For developers)\n"
        "  - Or pre-fetch this model once from an elevated (admin) shell\n"
        "Already-cached models (e.g. the default large-v3) are unaffected."
    )


def _load_langdetect():
    """Import the promoted shared language-detection helpers (sibling to the diarization clone).
    transcribe.py lives at transcribe-audio/scripts/, so the shared dir is parents[2]/_shared/..."""
    import importlib
    shared = Path(__file__).resolve().parents[2] / "_shared" / "langdetect" / "scripts"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    return importlib.import_module("language_detection")


# Re-export the shared helpers at module level so existing call sites + tests (tx.<name>) resolve
# unchanged. Bodies now live once in _shared/langdetect (were duplicated + drifting across skills).
_ld = _load_langdetect()
language_detection_windows = _ld.language_detection_windows
pick_detected_language = _ld.pick_detected_language
resolve_language = _ld.resolve_language


def transcribe(audio_path, output_dir=None, model_name="large-v3", language=None, diarize=True, formats=("txt", "srt", "json")):
    import torch
    import whisperx

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        sys.exit(f"Error: audio file not found: {audio_path}")

    output_dir = Path(output_dir).resolve() if output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        compute_type = "float16" if major >= 7 else "int8"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = 16 if vram_gb >= 12 else (8 if vram_gb >= 10 else 4)
    else:
        compute_type = "int8"
        batch_size = 4
    print(f"[transcribe] device={device} compute_type={compute_type} batch_size={batch_size} model={model_name}", file=sys.stderr)

    print(f"[transcribe] loading audio: {audio_path.name}", file=sys.stderr)
    audio = whisperx.load_audio(str(audio_path))
    duration_min = len(audio) / 16000 / 60
    print(f"[transcribe] audio length: {duration_min:.2f} min", file=sys.stderr)

    print(f"[transcribe] loading ASR model (first run downloads ~3GB)...", file=sys.stderr)
    try:
        asr = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    except OSError as e:
        hint = symlink_privilege_hint(e, model_name)
        if hint is None:
            raise
        sys.exit(hint)
    if language is None:
        language = resolve_language(asr, audio, prefix="transcribe")
    print(f"[transcribe] running ASR...", file=sys.stderr)
    result = asr.transcribe(audio, batch_size=batch_size, language=language)
    detected_lang = result.get("language", "unknown")
    print(f"[transcribe] detected language: {detected_lang}", file=sys.stderr)

    print(f"[transcribe] aligning word-level timestamps...", file=sys.stderr)
    try:
        align_model, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
    except Exception as e:
        print(f"[transcribe] alignment skipped: {e}", file=sys.stderr)

    if diarize:
        print(f"[transcribe] running speaker diarization (local models, no token)...", file=sys.stderr)
        try:
            # Diarization clone promoted to a shared location (A1.0 clone-seam decision). Explicit
            # sys.path insert keeps the bare `from diarize_pipeline import ...` working from either skill.
            import sys as _sys
            from pathlib import Path as _Path
            _shared = _Path(__file__).resolve().parents[2] / "_shared" / "diarization" / "scripts"
            if str(_shared) not in _sys.path:
                _sys.path.insert(0, str(_shared))
            from diarize_pipeline import get_pipeline, annotation_to_dataframe
            diarize_model = get_pipeline(device=device)
            waveform = torch.from_numpy(audio).unsqueeze(0)
            diarization = diarize_model({"waveform": waveform, "sample_rate": 16000})
            diarize_segments = annotation_to_dataframe(diarization)
        except RuntimeError as e:
            sys.exit(f"Error during diarization: {e}\nPass --no-diarize to skip speaker labeling.")
        result = whisperx.assign_word_speakers(diarize_segments, result)

    base = audio_path.stem
    segments = result.get("segments", [])
    written = []

    if "txt" in formats:
        txt_path = output_dir / f"{base}.txt"
        write_txt(segments, txt_path)
        written.append(txt_path)
    if "srt" in formats:
        srt_path = output_dir / f"{base}.srt"
        write_srt(segments, srt_path)
        written.append(srt_path)
    if "json" in formats:
        json_path = output_dir / f"{base}.json"
        json_path.write_text(json_mod.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        written.append(json_path)

    for p in written:
        print(f"[transcribe] wrote: {p}", file=sys.stderr)
    print(f"[transcribe] segments: {len(segments)}", file=sys.stderr)

    if written:
        sys.stdout.write(written[0].read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization (WhisperX wrapper)."
    )
    ap.add_argument("audio", help="Audio file path (.m4a, .mp3, .wav, etc.)")
    ap.add_argument("--output-dir", help="Output directory (default: same as audio)")
    ap.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
    ap.add_argument("--language", help="Force language code (e.g. 'en'). Auto-detect if omitted.")
    ap.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization")
    ap.add_argument("--format", choices=["txt", "srt", "json", "all"], default="all",
                    help="Output format(s). Default: all (writes .txt, .srt, .json)")
    args = ap.parse_args()

    formats = ("txt", "srt", "json") if args.format == "all" else (args.format,)

    transcribe(
        args.audio,
        output_dir=args.output_dir,
        model_name=args.model,
        language=args.language,
        diarize=not args.no_diarize,
        formats=formats,
    )


if __name__ == "__main__":
    main()
