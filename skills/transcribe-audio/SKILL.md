---
name: transcribe-audio
description: Transcribe audio files (m4a, mp3, wav, ogg, flac, webm, mp4) to text with speaker diarization using WhisperX. Use whenever the user provides an audio file path, asks to "transcribe", "convert audio to text", or wants a "transcript" of any recording. Outputs speaker-labeled txt, srt subtitles, and structured json with word-level timestamps. Triggers on .m4a, .mp3, .wav, .ogg, .flac, .webm, .mp4 paths in user messages.
---

# Audio Transcription (WhisperX)

## When to use

Any user request involving an audio file becoming text. Trigger phrases: "transcribe", "transcript of", "what does this audio say", "convert to text", or simply pasting a path ending in `.m4a`, `.mp3`, `.wav`, `.ogg`, `.flac`, `.webm`, `.mp4`.

## How to invoke

```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/transcribe-audio/scripts/transcribe.py "PATH_TO_AUDIO"
```

Or via the alias from any directory:
```bash
transcribe "PATH_TO_AUDIO"           # bash / git-bash
transcribe.bat "PATH_TO_AUDIO"       # cmd
```

## Outputs

Written next to the audio file by default:
- `{basename}.txt` — speaker-labeled plain text (`[SPEAKER_00] Hello...`)
- `{basename}.srt` — subtitles with timestamps for video editing
- `{basename}.json` — structured data with word-level timing and speaker labels

## Options

| Flag | Default | Purpose |
|---|---|---|
| `--output-dir DIR` | audio's folder | Write outputs elsewhere |
| `--format FMT` | `all` | Pick `txt`, `srt`, `json`, or `all` (writes all three) |
| `--model NAME` | `large-v3` | Override Whisper model |
| `--language CODE` | auto-detect | Force language (`en`, `es`, `fr`, etc.) |
| `--no-diarize` | off | Skip speaker labeling for speed |

Alternative models: `large-v3-turbo` (~3x faster, slightly less accurate on noisy audio), `medium`, `small`, `base`, `tiny`.

## Standalone diarization (no transcription)

For speaker segments only ("who spoke when", "diarize this", "speaker timeline"):

```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/_shared/diarization/scripts/diarize.py "PATH_TO_AUDIO"
```

Writes `{basename}.rttm` and `{basename}.json` next to the input. Flags: `--output-dir DIR`, `--num-speakers N` (or `--min-speakers`/`--max-speakers`), `--device cuda|cpu`, `--format rttm|json|all`. stdout carries the last-written format's content (json under the default `--format all`). No HF account or token: models fetch automatically on first use, verified against official checksums.

## Requirements (already installed)

- `whisperx` Python package
- `ffmpeg` 8.1 binary (on machine PATH via winget)
- `torch 2.8.0+cu126` (GPU auto-detected (CPU fallback))
- Diarization models (~32MB) auto-download on first use from token-free sources, sha256-verified against official checksums; cached in the shared clone at `_shared/diarization/models/diarization/`, fully offline afterward

## Performance

- GPU: ~5-15x realtime depending on card and diarization
- CPU fallback: ~2-5x realtime — 10 min audio in 2-5 min
- First run downloads ~1.5 GB of models (large-v3 + alignment) plus ~32MB of diarization weights (no account needed), then cached

## Troubleshooting

- **`OSError: [WinError 1314] A required privilege is not held by the client`** while
  loading a `--model` not yet in the local cache: Windows denied symlink creation in the
  huggingface cache, and huggingface_hub's threaded download raced past its copy-mode
  fallback. **Re-run first — it usually completes** (finished files are kept). Permanent
  fix: enable Windows Developer Mode (Settings > Update & Security > For developers), or
  pre-fetch the model once from an elevated shell. Already-cached models — including the
  default `large-v3` — are unaffected. `transcribe.py` detects this case and prints this
  same guidance.
- Diarization model fetch errors print the sources tried and the expected sha256; they
  never require an HF account — reconnect and re-run, verification is automatic.

## When NOT to use

- Skip if the user wants only audio analysis (tone, music identification) — Whisper is text-only
- Skip if the user provides a YouTube URL — download audio first via `yt-dlp`, then transcribe
- For non-speech audio (music, sound effects), output will be poor — flag this to the user before running
