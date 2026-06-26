"""Shared speech-window language detection for the transcription skills.

Pure, dependency-free decision logic (language_detection_windows, pick_detected_language)
plus the model-bound resolver (resolve_language). Promoted out of transcribe-video and
transcribe-audio, which carried byte-duplicated copies that had already drifted (divergent
stderr prefixes + docstrings). Each skill loads this via a _load_langdetect() path-insert
shim mirroring the _shared/diarization clone, and re-exports the names so call sites and
tests (tx.<name>) keep resolving unchanged.

Round-9 background: whisperx auto-detects language on the first ~30 s of raw audio; on a
silent / title-card intro that yields a confident-wrong language and a junk transcript
(a solo clip transcribed entirely as Welsh). These helpers always sample PAST t=0 and keep
the highest-confidence speech-bearing window, falling back to a default below a confidence floor."""
import sys


def language_detection_windows(duration_s, window_s=30.0):
    """Windows (start_s, end_s) to run language detection on. The round-9 garbage-language bug came
    from whisperx auto-detecting on the first ~30 s; on a silent / title-card intro that yields a
    confident-wrong language and a junk transcript. Unlike sample_windows (a single whole-clip
    window under 2 min, starting at t=0), this ALWAYS samples PAST the start, so a leading-silence
    intro can never be the only thing detected on. The caller detects on each window and keeps the
    highest-confidence result (pick_detected_language), so a window landing on silence self-eliminates."""
    duration_s = float(duration_s)
    if duration_s <= window_s:
        return [(0.0, duration_s)]  # too short to spread; the whole clip is all we have
    fracs = (0.10, 0.50, 0.90) if duration_s >= 120.0 else (0.35, 0.65)
    out = []
    for frac in fracs:
        center = duration_s * frac
        a = max(0.0, center - window_s / 2)
        b = min(duration_s, center + window_s / 2)
        out.append((a, b))
    return out


def pick_detected_language(candidates, default="en", min_confidence=0.5):
    """Choose the best (lang, probability) across detection windows -> (language, confidence,
    used_fallback). Picks the highest-probability candidate; if that best probability is below
    min_confidence (or there are no candidates) returns (default, best_conf_or_0.0, True) so the
    caller can warn loudly. This is the guard that stops a low-confidence detection (round-9 saw
    cy@0.27 and a lucky en@0.22) from poisoning the whole transcript."""
    best = max(candidates, key=lambda c: c[1], default=None)
    if best is None:
        return default, 0.0, True
    lang, conf = best
    if conf < min_confidence:
        return default, conf, True
    return lang, conf, False


def resolve_language(asr, audio, default="en", prefix="asr"):
    """Resolve the transcription language from SPEECH-bearing windows instead of whisperx's default
    first-30s-of-raw-audio detection, which mis-fires on a silent / title-card intro (the round-9
    garbage-Welsh failure). Runs faster-whisper language detection (which returns a probability,
    unlike whisperx's wrapper) on each language_detection_windows() slice; pick_detected_language
    keeps the highest-confidence result and falls back to `default` (with a loud warning) when even
    the best window is unsure. Model-bound; the decision logic it relies on is unit-tested.

    `prefix` tags the stderr lines per skill ([asr] for transcribe-video, [transcribe] for
    transcribe-audio) so promotion to this shared module preserves each skill's log text."""
    duration_s = len(audio) / 16000.0
    candidates = []
    for (a, b) in language_detection_windows(duration_s):
        clip = audio[int(a * 16000):int(b * 16000)]
        try:
            lang, prob, _ = asr.model.detect_language(clip)
        except Exception as e:
            print(f"[{prefix}] language detect failed on window {a:.0f}-{b:.0f}s: {e}", file=sys.stderr)
            continue
        candidates.append((lang, prob))
    lang, conf, used_fallback = pick_detected_language(candidates, default=default)
    if used_fallback:
        print(f"[{prefix}] language detection unreliable (best conf {conf:.2f}); falling back to '{lang}'. "
              f"Pass --language to override.", file=sys.stderr)
    else:
        print(f"[{prefix}] detected language '{lang}' (conf {conf:.2f}) from a speech-bearing window.",
              file=sys.stderr)
    return lang
