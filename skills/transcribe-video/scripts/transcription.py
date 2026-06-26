"""WhisperX transcription wrapper + §4.2 diarize-auto sample pass + audio analysis.

Pure decision functions (decide_diarization, count_speakers_over_floor,
estimate_speaker_count, sample_windows, parse_silence_gaps) are GPU-free and
unit-tested directly. The actual model calls (transcribe_segments, run_sample_pass)
run only in the gated e2e test."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# ---- pure, unit-testable decision logic ----

def auto_compute_type(gpu_name) -> str:
    """Pascal (GTX 10xx) → int8 for memory+speed; modern → float16; unknown/cpu → int8."""
    if not gpu_name:
        return "int8"
    name = gpu_name.upper()
    if "GTX 10" in name or "PASCAL" in name or "GTX 9" in name:
        return "int8"
    return "float16"


def auto_batch_size(vram_gb) -> int:
    """WhisperX ASR batch size scaled to GPU VRAM, mirroring transcribe-audio's proven logic.
    whisperx defaults to 16, which OOMs int8 large-v3 on an 8 GB Pascal card (GTX 1080), the
    GPU this project targets. batch_size is independent of clip length, so the default OOMs even
    short videos. cpu / unknown VRAM → conservative 4."""
    if not vram_gb:
        return 4
    if vram_gb >= 12:
        return 16
    if vram_gb >= 10:
        return 8
    return 4


def sample_windows(duration_s, window_s=30.0):
    """Up to three windows at ~10/50/90% of the timeline. <2 min → the whole clip."""
    if duration_s < 120.0:
        return [(0.0, float(duration_s))]
    out = []
    for frac in (0.10, 0.50, 0.90):
        center = duration_s * frac
        a = max(0.0, center - window_s / 2)
        b = min(duration_s, center + window_s / 2)
        out.append((a, b))
    return out


def _load_langdetect():
    """Import the promoted shared language-detection helpers (sibling to the diarization clone).
    transcription.py lives at transcribe-video/scripts/, so the shared dir is parents[2]/_shared/..."""
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


def count_speakers_over_floor(seconds_per_speaker: dict, min_speaker_seconds: float) -> int:
    """Distinct speakers with >= floor seconds of attributed speech (noise/overlap filtered)."""
    return sum(1 for s in seconds_per_speaker.values() if s >= min_speaker_seconds)


def estimate_speaker_count(window_speaker_seconds: list, min_speaker_seconds: float) -> int:
    """Given one {speaker: seconds} dict per sampled window, estimate distinct speakers as the
    MAX over windows of speakers-above-floor (F4).

    Why max, not sum: each window is diarized independently, so pyannote's per-window labels are
    NOT globally consistent — a solo speaker can be SPEAKER_00 in window 1 and SPEAKER_00 again in
    window 2, but summing labels across windows would also miscount if labels happened to differ.
    For the only decision we need (1 vs >=2), the max-within-any-window count is the correct,
    label-identity-independent signal: solo recordings show 1 in every window."""
    if not window_speaker_seconds:
        return 0
    return max(count_speakers_over_floor(w, min_speaker_seconds) for w in window_speaker_seconds)


_SIL_START_RE = re.compile(r"silence_start:\s*([0-9]+\.?[0-9]*)")
_SIL_END_RE = re.compile(r"silence_end:\s*([0-9]+\.?[0-9]*)")


def parse_silence_gaps(stderr_text: str) -> list:
    """Parse ffmpeg silencedetect stderr → midpoint of each silence interval (§4.5 #3 anchor).
    Hardened like parse_scene_times: returns [] on unexpected/empty output."""
    try:
        starts = [float(m.group(1)) for m in _SIL_START_RE.finditer(stderr_text or "")]
        ends = [float(m.group(1)) for m in _SIL_END_RE.finditer(stderr_text or "")]
    except (ValueError, TypeError):
        return []
    return [(s + e) / 2.0 for s, e in zip(starts, ends)]


def decide_diarization(flag: str, distinct_speakers: int, sample_errored: bool):
    """Map (flag, sample result) → (diarize_bool, diarization_reason enum). §4.2 + §16.4."""
    if flag == "on":
        return True, "forced_on"
    if flag == "off":
        return False, "forced_off"
    # auto
    if sample_errored or distinct_speakers == 0:
        return True, "auto_sample_error_failsafe_on"   # fail-safe ON
    if distinct_speakers >= 2:
        return True, "auto_multi_speaker"
    return False, "auto_single_speaker"


def speaker_turns(segments) -> list:
    """Timestamps where the speaker label changes (anchor signal, §4.5). Empty on captions."""
    turns = []
    prev = None
    for seg in segments:
        spk = seg.get("speaker")
        if prev is not None and spk != prev:
            turns.append(float(seg["start_s"]))
        prev = spk
    return turns


# ---- GPU / model calls (exercised only in the gated e2e test) ----

def detect_gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def detect_vram_gb():
    """Total VRAM of the active CUDA device in GB, or None if no GPU. Cheap (a property read,
    microseconds), so it is safe to call per launch to scale batch size to the installed card."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return None


def detect_silence_gaps(audio_path, noise_db=-30.0, min_silence_s=0.5):
    """Run ffmpeg silencedetect on the extracted audio; return silence-interval midpoints (§4.5 #3).
    Cheap single pass; only called on the WhisperX path where audio exists."""
    proc = subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence_s}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return parse_silence_gaps(proc.stderr)


def _load_clone():
    """Import the promoted token-free diarization clone (Task 6) from the shared location.
    transcription.py lives at transcribe-video/scripts/, so the shared dir is parents[2]/_shared/..."""
    import importlib
    shared = Path(__file__).resolve().parents[2] / "_shared" / "diarization" / "scripts"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    return importlib.import_module("diarize_pipeline")


def run_sample_pass(audio_path, duration_s, device, min_speaker_seconds=3.0):
    """§4.2 diarize-auto sample pass via the token-free clone (Task 6). CROPS each sample window to a
    short 16 kHz mono wav and diarizes ONLY that clip (F4) — total cost ~3×30 s regardless of file
    length. Aggregates with estimate_speaker_count (max over windows; per-clip pyannote labels are not
    globally consistent). Returns **(distinct_speakers, errored, weights_available)**. **No HF_TOKEN** —
    the clone fetches + sha256-verifies its weights itself and runs offline once cached.

    Weights-vs-runtime split: ensure_models() is called FIRST. If it can't obtain verified weights
    (cold offline first run) it raises, and we return weights_available=False so the run degrades
    diarization to OFF (reason auto_degraded_weights_unavailable) instead of failsafe-ON into a
    diarization that also can't load. Any OTHER failure in the actual pass (e.g. CUDA OOM) returns
    errored=True with weights_available=True → failsafe ON, the safe default for an ambiguous sample."""
    import torch
    import whisperx
    clone = _load_clone()
    try:
        clone.ensure_models()  # fetch + sha256-verify weights up front; RuntimeError if unobtainable
    except Exception as e:
        print(f"[diarize-auto] diarization weights unavailable, degrading to off: {e}", file=sys.stderr)
        return 0, False, False
    clips = []
    try:
        pipe = clone.get_pipeline(device=device)
        window_seconds = []
        for (a, b) in sample_windows(duration_s):
            clip = Path(tempfile.gettempdir()) / f"a1_0_sample_{int(a)}_{int(b)}.wav"
            clips.append(clip)
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(audio_path),
                 "-ac", "1", "-ar", "16000", str(clip)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            # clone pipeline takes {"waveform": (1, T) float32 tensor, "sample_rate": 16000}
            wav = torch.from_numpy(whisperx.load_audio(str(clip))).unsqueeze(0)
            out = pipe({"waveform": wav, "sample_rate": 16000})
            annotation = clone.unwrap_annotation(out)  # pyannote 4.x DiarizeOutput → Annotation
            seconds = {}
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                seconds[speaker] = seconds.get(speaker, 0.0) + (turn.end - turn.start)
            window_seconds.append(seconds)
        return estimate_speaker_count(window_seconds, min_speaker_seconds), False, True
    except Exception as e:
        print(f"[diarize-auto] sample pass error, failing safe to ON: {e}", file=sys.stderr)
        return 0, True, True
    finally:
        for clip in clips:
            try:
                clip.unlink()
            except OSError:
                pass


def transcribe_segments(audio_path, model_name, language, diarize, device, compute_type):
    """WhisperX transcribe (+ optional diarization). Returns SegmentRecord-shaped list. When
    `language` is None, resolves it from speech-bearing windows (resolve_language) rather than
    whisperx's silent-intro-prone first-30s auto-detect."""
    import whisperx
    audio = whisperx.load_audio(str(audio_path))
    print("[asr] loading ASR model (first run downloads ~3GB)...", file=sys.stderr)
    asr = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    batch_size = auto_batch_size(detect_vram_gb() if device == "cuda" else None)
    print(f"[asr] device={device} compute_type={compute_type} batch_size={batch_size}", file=sys.stderr)
    if language is None:
        language = resolve_language(asr, audio)
    result = asr.transcribe(audio, batch_size=batch_size, language=language)
    detected = result.get("language", "unknown")
    try:
        amodel, meta = whisperx.load_align_model(language_code=detected, device=device)
        result = whisperx.align(result["segments"], amodel, meta, audio, device, return_char_alignments=False)
    except Exception as e:
        print(f"[asr] word-align skipped: {e}", file=sys.stderr)
    if diarize:
        # Token-free clone (Task 6): NO HF_TOKEN, NO whisperx.diarize.DiarizationPipeline. The clone
        # pipeline consumes a {"waveform": (1,T) tensor, "sample_rate": 16000} dict; annotation_to_dataframe
        # yields the start/end/speaker frame whisperx.assign_word_speakers expects.
        import torch
        clone = _load_clone()
        pipe = clone.get_pipeline(device=device)
        wav = torch.from_numpy(audio).unsqueeze(0)  # whisperx.load_audio is 16 kHz mono float32
        diar_out = pipe({"waveform": wav, "sample_rate": 16000})
        diar_df = clone.annotation_to_dataframe(diar_out)
        result = whisperx.assign_word_speakers(diar_df, result)
    raw = result.get("segments", [])
    return _to_segment_records(raw), detected


def _to_segment_records(raw):
    segs = []
    for i, s in enumerate(raw):
        segs.append({
            "index": i, "start_s": float(s["start"]), "end_s": float(s["end"]),
            "speaker": s.get("speaker"), "text": (s.get("text") or "").strip(),
            "frame_index": None,
        })
    return segs
