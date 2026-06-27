# Architecture

Two skills over a shared, token-free diarization core.

## transcribe-audio

```
audio file
  -> ffmpeg decode (16 kHz mono)
  -> WhisperX (large-v3) ASR
  -> language detect on speech-bearing windows (skips silent intros)
  -> word-level alignment
  -> token-free diarization (speaker labels)
  -> write {name}.txt / .srt / .json
```

## transcribe-video (curate-and-stop)

```
video URL or file
  -> yt-dlp (URL) | direct (local file)
  -> captions present?  capture verbatim  :  extract audio -> WhisperX ASR (+ diarization)
  -> dense 1fps sample -> content-box -> segment into held scenes by phash@16 change
       -> keep each scene's first non-junk frame (scene-start); ffmpeg scene-cuts feed alignment anchors only
  -> joint-signal alignment (transcript segment <-> frame)
  -> write the 5-artifact curated set:
       {name}_transcript.txt, {name}_frames/, {name}_contactsheet.jpg,
       {name}_frames.md, {name}_manifest.json  (validates manifest-1.0.schema.json)
  -> STOP   (no composed guide; that is a later compose tier)
```

## Shared diarization

`skills/_shared/diarization/` is a token-free pyannote clone. It fetches and sha256-verifies its
own weights on first use (no HF token), then runs offline. Both skills resolve it via a relative
path (`parents[2]/_shared/...`), so mirroring the dev `skills/` layout inside the plugin makes it
work with zero code change.

## Runtime auto-detection

`compute_type` (float16 on modern GPUs, int8 on Pascal/CPU) and ASR `batch_size` scale to the
detected GPU's VRAM. No specific card is hard-coded - the same build runs on any NVIDIA GPU or CPU.
