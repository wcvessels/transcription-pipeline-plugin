# Troubleshooting (by symptom)

**`ffmpeg` not found / "ffmpeg is not recognized".**
Install it and restart the shell - PATH does not propagate to already-running shells.
Windows: `winget install Gyan.FFmpeg`.

**`RuntimeError: CUDA failed with error out of memory`.**
ASR batch size scales to VRAM automatically, but a large model on a small card can still OOM.
Use a smaller `--model` (e.g. `medium`) or run on CPU.

**`ValueError: Requested float16 compute type, but the target device ... do not support efficient float16`.**
Pascal GPUs (compute capability < 7) lack efficient fp16. The skills auto-pick `int8` for those;
if you still see this, the card was misdetected - report the GPU name from `check-environment.py`.

**Transcription is very slow.**
You are on CPU. `check-environment.py` will show "CUDA GPU: none". Install the CUDA build of torch
(see `/transcribe-setup`) or accept ~2-5x realtime on CPU.

**`OSError: [WinError 1314] A required privilege is not held` while downloading a model.**
Windows denied symlink creation in the HuggingFace cache. Re-run - it usually completes (finished
files are kept). Permanent fix: enable Developer Mode (Settings > Update & Security > For
developers), or pre-fetch the model once from an elevated shell. Already-cached models are fine.

**Wrong-language transcript (e.g. English audio transcribed as Welsh).**
Old failure on clips with a silent / title-card intro: detection ran on the silent first 30 s. The
skills now detect language on speech-bearing windows. If it still misfires on unusual audio, force
it with `--language en`.

**yt-dlp warns about YouTube extraction / no JS runtime.**
yt-dlp ages quickly. Update it: `python -m pip install -U yt-dlp`.

**Diarization weight-fetch errors.**
The clone prints the sources tried and the expected sha256; no account is needed. Reconnect and
re-run - verification is automatic.
