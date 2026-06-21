---
description: Guided one-time setup for the transcription plugin - Python deps, ffmpeg, GPU check.
---

Set this machine up to run the transcription plugin. It is token-free: no HuggingFace account or token is needed.

Run these steps in order, stopping to report any failure before continuing:

1. **Python** - confirm 3.10+ (`python --version`; 3.12 recommended).

2. **ffmpeg** - `ffmpeg -version`. If missing on Windows: `winget install Gyan.FFmpeg`, then tell the user to restart the shell (PATH does not propagate to the running session).

3. **PyTorch with CUDA** (NVIDIA GPU present) - install the CUDA build, NOT the default PyPI CPU build:
   `python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126`
   No NVIDIA GPU: `python -m pip install torch torchaudio` (CPU; transcription will be much slower).

4. **Remaining dependencies** - `python -m pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"`

5. **Verify** - `python "${CLAUDE_PLUGIN_ROOT}/scripts/check-environment.py"`. It must print `READY` (exit 0). ASR + diarization model weights (~3 GB WhisperX + ~32 MB diarization) download automatically on the first transcription, sha256-verified, no token.

6. **(Optional) terminal commands** - to run `transcribe` / `transcribe-video` from any terminal, add `${CLAUDE_PLUGIN_ROOT}/bin` to PATH. Otherwise just use the skills inside a Claude Code session - they auto-trigger on audio/video paths and URLs.

Report what was installed and the final check-environment result.
