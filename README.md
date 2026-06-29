# transcription (Claude Code plugin)

Local, token-free audio + video transcription with speaker diarization and screenshot curation.
Runs entirely on your machine - WhisperX large-v3 + a token-free pyannote clone. No HuggingFace
account or token.

- **transcribe-audio** - any audio file -> speaker-labeled `txt` + `srt` + `json`.
- **transcribe-video** - a video URL or file -> a curated set: transcript + per-scene
  screenshots (one per distinct on-screen scene, at its scene-start) + frames index + schema-validated manifest (the inputs a guide is composed from later).

Windows + NVIDIA GPU recommended; CPU works but is much slower. The GPU class is auto-detected.

## Install

Installing the plugin gives you the skills, the `/transcribe-setup` command, the `bin/` wrappers,
and docs. It does **not** install Python deps, ffmpeg, or models - run setup next.

1. **Marketplace:**
   ```
   /plugin marketplace add https://github.com/wcvessels/transcription-pipeline-plugin
   /plugin install transcription-pipeline
   ```
2. **Local clone:**
   ```
   git clone https://github.com/wcvessels/transcription-pipeline-plugin.git transcription-pipeline-plugin
   /plugin marketplace add ./transcription-pipeline-plugin
   /plugin install transcription-pipeline
   ```
3. **Manual** (no plugin system): copy `skills/*` into `~/.claude/skills/`.

## Setup (one-time)

In a Claude Code session:
```
/transcribe-setup
```
It checks Python + ffmpeg, installs PyTorch (CUDA build) + the deps, and runs the self-test. Model
weights (~3 GB WhisperX + ~32 MB diarization) download on the first transcription, sha256-verified,
no token. Self-test any time: `python scripts/check-environment.py`.

## Use

In a Claude Code session, just give a path or URL ("transcribe meeting.m4a", "make docs from
recording.mp4") - the skills auto-trigger.

From a terminal (add `bin/` to PATH first):
```
transcribe audio.m4a
transcribe-video https://youtube.com/watch?v=...
transcribe-video recording.mp4 --max-frames 50
```
Outputs land next to the input (or `--output-dir`). See each skill's `SKILL.md` for the full flags.

## Requirements

Python 3.10+ (3.12 recommended), ffmpeg on PATH, ~3.5 GB disk for models. NVIDIA GPU optional (CPU
fallback). Windows-only for v1.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - the pipeline stages.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - fixes by symptom.
- [CHANGELOG.md](CHANGELOG.md)

## Maintenance

The skills are a versioned snapshot of the dev source (`~/.claude/skills/`). Refresh with
`scripts/sync-from-dev.sh` and bump the version in `.claude-plugin/plugin.json`.

## License and credits

MIT (see [LICENSE](LICENSE)), Copyright (c) 2026 Will Vessels.

This tool downloads and uses third-party models and libraries that carry their own
licenses (the pyannote diarization weights, WhisperX, faster-whisper, PyTorch). No model
weights are bundled; they are auto-fetched and sha256-verified on first use. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for attributions and citations.
