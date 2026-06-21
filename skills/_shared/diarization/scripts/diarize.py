#!/usr/bin/env python
"""Standalone speaker diarization: audio in, speaker segments out. No HF token.

Writes {basename}.rttm and {basename}.json next to the input (or --output-dir).
Models auto-download on first use (~32MB, checksum-verified), then run offline.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from diarize_pipeline import get_pipeline, unwrap_annotation


def annotation_to_records(annotation):
    return [
        {"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": speaker}
        for seg, _, speaker in annotation.itertracks(yield_label=True)
    ]


def write_json(annotation, path):
    Path(path).write_text(
        json.dumps(annotation_to_records(annotation), indent=2), encoding="utf-8"
    )


def write_rttm(annotation, path):
    with open(path, "w", encoding="utf-8") as f:
        annotation.write_rttm(f)


def diarize(audio_path, output_dir=None, num_speakers=None, min_speakers=None,
            max_speakers=None, device=None, formats=("rttm", "json")):
    import torch
    import whisperx

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        sys.exit(f"Error: audio file not found: {audio_path}")
    output_dir = Path(output_dir).resolve() if output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diarize] loading audio: {audio_path.name}", file=sys.stderr)
    audio = whisperx.load_audio(str(audio_path))  # 16kHz mono float32 numpy
    print(f"[diarize] audio length: {len(audio) / 16000 / 60:.2f} min", file=sys.stderr)

    print(f"[diarize] loading pipeline (first run downloads ~32MB)...", file=sys.stderr)
    try:
        pipeline = get_pipeline(device=device)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")

    speaker_args = {}
    if num_speakers is not None:
        speaker_args["num_speakers"] = num_speakers
    if min_speakers is not None:
        speaker_args["min_speakers"] = min_speakers
    if max_speakers is not None:
        speaker_args["max_speakers"] = max_speakers

    print(f"[diarize] running diarization...", file=sys.stderr)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    result = pipeline({"waveform": waveform, "sample_rate": 16000}, **speaker_args)
    annotation = unwrap_annotation(result)  # 4.x returns DiarizeOutput
    annotation.uri = re.sub(r"\s+", "_", audio_path.stem)

    speakers = sorted(annotation.labels())
    msg = f"[diarize] found {len(speakers)} speakers" + (f": {', '.join(speakers)}" if speakers else "")
    print(msg, file=sys.stderr)

    base = audio_path.stem
    written = []
    if "rttm" in formats:
        rttm_path = output_dir / f"{base}.rttm"
        write_rttm(annotation, rttm_path)
        written.append(rttm_path)
    if "json" in formats:
        json_path = output_dir / f"{base}.json"
        write_json(annotation, json_path)
        written.append(json_path)

    for p in written:
        print(f"[diarize] wrote: {p}", file=sys.stderr)
    if written:
        sys.stdout.write(written[-1].read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(
        description="Speaker diarization with locally-verified pyannote models (no HF token)."
    )
    ap.add_argument("audio", help="Audio file path (.m4a, .mp3, .wav, etc.)")
    ap.add_argument("--output-dir", help="Output directory (default: same as audio)")
    ap.add_argument("--num-speakers", type=int, help="Exact speaker count, if known (overrides --min/--max-speakers)")
    ap.add_argument("--min-speakers", type=int, help="Lower bound on speaker count")
    ap.add_argument("--max-speakers", type=int, help="Upper bound on speaker count")
    ap.add_argument("--device", choices=["cuda", "cpu"], help="Default: cuda if available")
    ap.add_argument("--format", choices=["rttm", "json", "all"], default="all",
                    help="Output format(s). Default: all (writes .rttm and .json)")
    args = ap.parse_args()

    if args.num_speakers is not None and args.num_speakers < 1:
        ap.error("--num-speakers must be >= 1")
    if args.min_speakers is not None and args.max_speakers is not None and args.min_speakers > args.max_speakers:
        ap.error("--min-speakers cannot exceed --max-speakers")

    formats = ("rttm", "json") if args.format == "all" else (args.format,)
    diarize(
        args.audio,
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
        formats=formats,
    )


if __name__ == "__main__":
    main()
