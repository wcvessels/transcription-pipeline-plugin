# Changelog

## 1.2.0 — unreleased

- **transcribe-video frame-curation rewrite** — replaced best-of-window curation with **dense change-detection**: sample at 1 fps, segment the timeline into held on-screen scenes by perceptual-hash (phash@16) change between consecutive frames, and keep each scene's first non-junk frame (its scene-start). Captures every distinct on-screen scene; fixes a bug where smooth in-app transitions left content screens uncaptured. ffmpeg scene-cuts now feed alignment anchors only. Manifest schema unchanged (`curation` fields remapped).

## 1.0.0 — unreleased

Initial packaging of the transcription stack as a Claude Code plugin.

- **transcribe-audio** — WhisperX transcription + token-free diarization; outputs txt / srt / json.
- **transcribe-video** — curate-and-stop pipeline: speaker-labeled transcript + best-of-window deduped screenshots + contact sheet + frames index + schema-validated manifest. No composed guide (that is a later compose tier).
- **Language detection on speech-bearing windows** — skips silent / title-card intros that previously caused wrong-language transcripts.
- **Token-free** — no HuggingFace account or token; diarization weights are sha256-verified on first fetch, then offline.
- **/transcribe-setup** wizard + `scripts/check-environment.py` self-test.
- Windows + NVIDIA GPU, CPU fallback. GPU class (compute type, batch size) auto-detected at runtime.
