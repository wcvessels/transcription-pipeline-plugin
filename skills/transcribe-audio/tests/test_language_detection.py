"""Language-detection robustness for transcribe-audio (mirrors transcribe-video's fix).

Bug: --language defaulted to None -> whisperx auto-detected on the first ~30s of raw audio, so a
near-silent / title-card intro yielded a confident-wrong language and a garbage transcript. Fix:
detect on SPEECH-bearing windows and keep the highest-confidence result, falling back to a default
only when even the best window is unsure.
"""
import numpy as np

import transcribe as tx


class _ScriptedASR:
    """Minimal stand-in for a whisperx pipeline. resolve_language only touches
    `.model.detect_language(audio) -> (lang, prob, all_probs)`, so the fake IS its own model and
    returns scripted results in call order (one per detection window). No GPU, no model load."""
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0
        self.model = self

    def detect_language(self, audio):
        r = self._scripted[self.calls]
        self.calls += 1
        return r


def test_language_detection_windows_long_clip_spreads():
    wins = tx.language_detection_windows(duration_s=600.0, window_s=30.0)
    centers = [round((a + b) / 2) for a, b in wins]
    assert centers == [60, 300, 540]


def test_language_detection_windows_short_clip_skips_intro():
    wins = tx.language_detection_windows(duration_s=90.0, window_s=30.0)
    assert wins != [(0.0, 90.0)]
    assert len(wins) >= 2
    assert all(a > 0.0 for a, b in wins)


def test_language_detection_windows_tiny_clip_is_whole():
    assert tx.language_detection_windows(duration_s=20.0, window_s=30.0) == [(0.0, 20.0)]


def test_pick_language_takes_highest_confidence_window():
    cands = [("cy", 0.27), ("en", 0.93), ("en", 0.81)]
    assert tx.pick_detected_language(cands) == ("en", 0.93, False)


def test_pick_language_falls_back_below_confidence_floor():
    lang, conf, fb = tx.pick_detected_language([("cy", 0.27), ("en", 0.22)], default="en", min_confidence=0.5)
    assert lang == "en" and fb is True and conf == 0.27


def test_pick_language_empty_falls_back():
    assert tx.pick_detected_language([], default="en") == ("en", 0.0, True)


def test_resolve_language_uses_speech_window_over_silent_intro():
    audio = np.zeros(180 * 16000, dtype="float32")
    asr = _ScriptedASR([("cy", 0.27, []), ("en", 0.95, []), ("en", 0.90, [])])
    assert tx.resolve_language(asr, audio) == "en"
    assert asr.calls == 3


def test_resolve_language_falls_back_when_all_windows_unsure():
    audio = np.zeros(180 * 16000, dtype="float32")
    asr = _ScriptedASR([("cy", 0.27, []), ("en", 0.22, []), ("ru", 0.30, [])])
    assert tx.resolve_language(asr, audio, default="en") == "en"
