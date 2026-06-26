---
name: transcribe-video
description: Convert a video (URL or local file) into a curated artifact set — a speaker-labeled transcript, best-of-window deduped screenshots at scene-change moments, a contact sheet, and a structured manifest (the local inputs a guide or walkthrough is composed from later). Use when the user provides a video URL (YouTube, Vimeo, etc.) or a local video file (.mp4, .mov, .mkv, .webm, .avi) and wants a transcript, screenshots aligned to spoken content, or the source material for a guide or training documentation. Triggers on video file paths, video URLs, or phrases like "transcribe this video", "make a guide from", "turn this recording into docs".
---

# Video to curated artifact set (transcript + curated screenshots)

## When to use

Any video → curated transcript + screenshots workflow (the local inputs a guide or walkthrough is built from):
- Training session recording → speaker-labeled transcript + curated screenshots
- Tutorial video URL → transcript + aligned screenshots
- Meeting recording → transcript with speaker labels and visual context
- Self-recorded "sloppy" walkthroughs → clean curated source material
- Any video where you want both *what was said* and *what was shown*

Triggers: video URLs (youtube.com, vimeo.com, etc.), local paths ending in `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, or phrases like "transcribe this video", "make docs from this recording", "turn this into a guide".

## How to invoke

```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/transcribe-video/scripts/transcribe.py "URL_OR_PATH"
```

Or via the alias from any directory:
```bash
transcribe-video "URL_OR_PATH"        # bash / git-bash
transcribe-video.bat "URL_OR_PATH"    # cmd
```

## What it does

1. **Detects input** — URL → downloads via `yt-dlp` (with captions if available); local file → uses directly
2. **Caption-vs-transcribe** — if the source has captions, captures them verbatim (no audio extraction); otherwise extracts audio and transcribes with WhisperX (large-v3)
3. **Token-free diarization** — speaker labels via the shared pyannote clone (no HF_TOKEN); `--diarize auto` runs a bounded sample pass to decide
4. **Detects scene changes** via ffmpeg's scene filter (default threshold 0.3)
5. **Best-of-window frame curation** — samples candidates per scene-cut settle window, scores sharpness + information, keeps the best non-junk, then dedups near-duplicates with a joint phash+colorhash gate
6. **Joint-signal alignment** — maps each transcript segment to a frame by timestamp (writes `segments[].frame_index`)
7. **Writes the curated artifact set and stops** — no guide is composed (that is a later compose tier)

## Outputs (next to source video, or in `--output-dir`) — curated artifact set

- `{basename}_frames/` — curated screenshots (`frame_0001_001530.jpg …`, index + `HHMMSS` timestamp; best-of-window + joint phash+colorhash deduped)
- `{basename}_transcript.txt` — verbatim transcript (speaker-grouped on the WhisperX path; plain on the captions path) — written on **both** paths
- `{basename}_manifest.json` — structured manifest, validates against `manifest-1.0.schema.json`
- `{basename}_contactsheet.jpg` — thumbnail grid of the curated frames, each captioned with its timestamp
- `{basename}_frames.md` — flat frames index: timestamp + sharpness + image link per kept frame

A1.0 **curates and stops** — it produces these local inputs for the prosumer compose tier; it does NOT write a guide.

## Options

| Flag | Default | Purpose |
|---|---|---|
| `--output-dir DIR` | source's folder | where everything goes |
| `--scene-threshold X` | `0.3` | ffmpeg scene-detection sensitivity (lower = more frames) |
| `--max-frames N` | `100` | cap on extracted screenshots |
| `--interval-seconds N` | off | fixed-interval frames instead of scene detection |
| `--frames-per-minute N` | `5` | target frame cadence when scene-detect finds nothing |
| `--window-size N` | `5` | best-of-window: candidates sampled per settle window; `1` = escape hatch (plain extract-then-dedup) |
| `--dedup-threshold N` | `5` | phash Hamming distance for the joint phash+colorhash dedup gate; `0` disables all dedup |
| `--allow-low-quality-frames` | off | if every frame scores as junk, keep the single best candidate instead of failing (accepted lower fidelity) |
| `--model NAME` | `large-v3` | Whisper model |
| `--language CODE` | auto | force language code |
| `--diarize {auto,on,off}` | `auto` | speaker diarization (token-free, no HF_TOKEN); `auto` runs a bounded sample pass |
| `--force-transcribe` | env¹ | run WhisperX even when captions exist (overrides the env default) |
| `--prefer-captions` | env¹ | force the captions path when captions exist (overrides the env default; inverse of `--force-transcribe`) |
| `--keep-audio` | off | keep extracted `audio.wav` |
| `--keep-work` | off | keep the `{basename}_work/` scratch dir |
| `--source-hint {url,file,…}` | auto | force source branch; connector values are reserved (A1.x, error) |

¹ Caption-vs-transcribe default: env `TRANSCRIBE_VIDEO_FORCE_TRANSCRIBE` (`1`/`true`/`yes`/`on` → transcribe; default captions-first). Precedence: flag > env > captions-first; `--force-transcribe` and `--prefer-captions` are mutually exclusive.

## Common workflows

```bash
# Training video → curated transcript + screenshots
transcribe-video "C:/Recordings/training-2026-04.mp4"

# YouTube tutorial → transcript + aligned screenshots
transcribe-video "https://youtube.com/watch?v=..."

# Long meeting with many scene changes — cap frames
transcribe-video "meeting.mkv" --max-frames 50

# Slide-deck recording — fewer scene changes, lower threshold
transcribe-video "deck-walkthrough.mp4" --scene-threshold 0.15
```

## Requirements (already installed)

- `whisperx`, `faster-whisper`, `pyannote-audio`
- `ffmpeg` 8.1 on PATH
- `yt-dlp` (for URL ingestion)
- `torch` with CUDA 12.6 (GPU auto-detected (CPU fallback))
- No `HF_TOKEN` and no pyannote license — diarization weights are sha256-verified public-mirror fetches, cached and offline-ready after first use

## Performance

- 10-min video → curated set: ~1-2 min on GPU, ~5-10 min on CPU
- First run downloads ~3 GB of WhisperX models (cached after)
- Disk: temp audio + frames; cleaned automatically unless `--keep-audio` or `--keep-work`

## When NOT to use

- Pure audio with no visual changes → use `transcribe-audio` instead (no need to extract frames)
- Animation-heavy or game footage where scene detection misfires constantly — use `--interval-seconds`
- Live streams or partial downloads — yt-dlp will refuse; download full video first
