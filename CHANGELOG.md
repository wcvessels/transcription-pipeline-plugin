# Changelog

## 1.2.2 — unreleased

- **transcribe-video output leanness (A1.2 Shape-1)** — scrapped the contact sheet entirely; every run now emits a lean 4-artifact set (`{name}_transcript.txt`, `{name}_frames/`, `{name}_frames.md`, `{name}_manifest.json`). Manifest **schema 1.1** (= 1.0 minus `contactsheet_jpg`; `frames_index_md` is now a required string) is the new canonical; `manifest-1.0.schema.json` stays frozen and downstream readers accept either via `schema_version`. Retired 3 dead no-op flags (`--allow-low-quality-frames`, `--frames-per-minute`, `--window-size`). Plus a curation coverage-invariant fix (never drop an all-junk scene) and a content-box thumbnail perf improvement. 150 (video) + 8 (audio) tests green.

## 1.2.1 — unreleased

- **transcribe-video curation review fixes** — folded external Codex + Gemini adversarial reviews of the change-detection rewrite. Segmentation now compares each frame to its **scene anchor** (not the previous frame), so a slow transition between two distinct screens splits instead of merging; the kept frame's **image** (first clear frame) and **timestamp** (scene start) are decoupled, fixing scene-cut / narration alignment on slow transitions. Correctness fixes: the duration cap now preserves the video's final scene (was tail-truncating), the content-box trim is computed over all sampled frames (was a 24-frame subsample), junk-scoring uses the content box, and scene-cut times are zero-based to the frame clock. Manifest schema unchanged. 149 (video) + 8 (audio) tests green.
- **Diarization mirror provenance** — `SEGMENTATION_SOURCES` reordered license-compliant-first (tensorlake → ubitec → ivrit-ai); all sha256-pinned to the identical file.

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
