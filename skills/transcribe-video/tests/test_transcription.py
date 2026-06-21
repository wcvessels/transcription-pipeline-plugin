import numpy as np

import transcription as tx


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


def test_resolve_language_uses_speech_window_over_silent_intro():
    # 180 s -> three windows (10/50/90%). The intro window detects garbage at low prob (cy@0.27);
    # the speech windows detect en confidently. The whole point: it does NOT stop at the first 30 s.
    audio = np.zeros(180 * 16000, dtype="float32")
    asr = _ScriptedASR([("cy", 0.27, []), ("en", 0.95, []), ("en", 0.90, [])])
    assert tx.resolve_language(asr, audio) == "en"
    assert asr.calls == 3  # detected on every window, not only the leading 30 s


def test_resolve_language_falls_back_when_all_windows_unsure():
    audio = np.zeros(180 * 16000, dtype="float32")
    asr = _ScriptedASR([("cy", 0.27, []), ("en", 0.22, []), ("ru", 0.30, [])])
    assert tx.resolve_language(asr, audio, default="en") == "en"  # below floor -> default


def test_auto_compute_type_pascal_is_int8():
    assert tx.auto_compute_type("NVIDIA GeForce GTX 1080") == "int8"
    assert tx.auto_compute_type("NVIDIA GeForce RTX 4090") == "float16"
    assert tx.auto_compute_type(None) == "int8"  # cpu / unknown → conservative


def test_auto_batch_size_scales_with_vram():
    # mirror transcribe-audio's proven thresholds. whisperx defaults to batch_size=16, which OOMs
    # int8 large-v3 on an 8 GB Pascal card (GTX 1080), the GPU this project targets.
    assert tx.auto_batch_size(24.0) == 16   # >=12 GB (e.g. RTX 4090)
    assert tx.auto_batch_size(11.0) == 8    # 10-12 GB
    assert tx.auto_batch_size(8.59) == 4    # GTX 1080, 8 GB → must NOT be 16
    assert tx.auto_batch_size(None) == 4    # cpu / unknown → conservative


def test_sample_window_centers_for_long_clip():
    # >=2 min → three windows at ~10/50/90% of duration
    wins = tx.sample_windows(duration_s=600.0, window_s=30.0)
    centers = [round((a + b) / 2) for a, b in wins]
    assert centers == [60, 300, 540]


def test_sample_window_short_clip_is_whole():
    wins = tx.sample_windows(duration_s=90.0, window_s=30.0)
    assert wins == [(0.0, 90.0)]


def test_language_detection_windows_long_clip_spreads():
    # >=2 min → sample at 10/50/90% so a silent intro is never the only thing detected on
    wins = tx.language_detection_windows(duration_s=600.0, window_s=30.0)
    centers = [round((a + b) / 2) for a, b in wins]
    assert centers == [60, 300, 540]


def test_language_detection_windows_short_clip_skips_intro():
    # the bug: a silent / title-card intro at t=0. Unlike sample_windows (which returns a single
    # whole-clip window under 2 min), a short clip must sample PAST the intro, never start at 0.
    wins = tx.language_detection_windows(duration_s=90.0, window_s=30.0)
    assert wins != [(0.0, 90.0)]
    assert len(wins) >= 2
    assert all(a > 0.0 for a, b in wins)  # nothing begins at the very start


def test_language_detection_windows_tiny_clip_is_whole():
    # below one window length there is nothing to spread; the whole clip is all we have
    assert tx.language_detection_windows(duration_s=20.0, window_s=30.0) == [(0.0, 20.0)]


def test_pick_language_takes_highest_confidence_window():
    # the silent-intro window detects garbage at low prob (round-9: cy@0.27); a speech window wins
    cands = [("cy", 0.27), ("en", 0.93), ("en", 0.81)]
    assert tx.pick_detected_language(cands) == ("en", 0.93, False)


def test_pick_language_falls_back_below_confidence_floor():
    # every window is low-confidence (mostly-silent clip) → fall back to default + flag for a warning.
    # round-9 actuals: a wrong cy@0.27 and a lucky en@0.22, both should be treated as unreliable.
    lang, conf, fb = tx.pick_detected_language([("cy", 0.27), ("en", 0.22)], default="en", min_confidence=0.5)
    assert lang == "en" and fb is True and conf == 0.27


def test_pick_language_empty_falls_back():
    assert tx.pick_detected_language([], default="en") == ("en", 0.0, True)


def test_decide_auto_single_speaker():
    diar, reason = tx.decide_diarization("auto", distinct_speakers=1, sample_errored=False)
    assert diar is False and reason == "auto_single_speaker"


def test_decide_auto_multi_speaker():
    diar, reason = tx.decide_diarization("auto", distinct_speakers=3, sample_errored=False)
    assert diar is True and reason == "auto_multi_speaker"


def test_decide_auto_failsafe_on_error_or_zero():
    assert tx.decide_diarization("auto", distinct_speakers=0, sample_errored=False) == (True, "auto_sample_error_failsafe_on")
    assert tx.decide_diarization("auto", distinct_speakers=2, sample_errored=True) == (True, "auto_sample_error_failsafe_on")


def test_decide_forced_modes():
    assert tx.decide_diarization("on", distinct_speakers=1, sample_errored=False) == (True, "forced_on")
    assert tx.decide_diarization("off", distinct_speakers=3, sample_errored=False) == (False, "forced_off")


def test_count_speakers_over_floor():
    # per-speaker attributed seconds within one window; floor = 3.0s
    seconds = {"SPEAKER_00": 25.0, "SPEAKER_01": 1.5, "SPEAKER_02": 8.0}
    assert tx.count_speakers_over_floor(seconds, min_speaker_seconds=3.0) == 2  # 01 is below floor


def test_estimate_speaker_count_uses_max_over_windows():
    # F4: solo speaker re-labeled per window must NOT be summed into a false multi-speaker count
    solo = [{"SPEAKER_00": 28.0}, {"SPEAKER_00": 30.0}, {"SPEAKER_00": 27.0}]
    assert tx.estimate_speaker_count(solo, min_speaker_seconds=3.0) == 1
    # any window showing two speakers → multi
    multi = [{"SPEAKER_00": 28.0}, {"SPEAKER_00": 15.0, "SPEAKER_01": 12.0}, {"SPEAKER_00": 27.0}]
    assert tx.estimate_speaker_count(multi, min_speaker_seconds=3.0) == 2
    assert tx.estimate_speaker_count([], min_speaker_seconds=3.0) == 0


def test_parse_silence_gaps_midpoints():
    stderr = ("[silencedetect @ 0x0] silence_start: 4.0\n"
              "[silencedetect @ 0x0] silence_end: 6.0 | silence_duration: 2.0\n"
              "[silencedetect @ 0x0] silence_start: 10.0\n"
              "[silencedetect @ 0x0] silence_end: 11.0 | silence_duration: 1.0\n")
    assert tx.parse_silence_gaps(stderr) == [5.0, 10.5]


def test_parse_silence_gaps_tolerates_garbage():
    assert tx.parse_silence_gaps("") == []
    assert tx.parse_silence_gaps("ffmpeg version 8.1\nno silence here") == []


def test_speaker_turns_from_segments():
    segs = [
        {"start_s": 0.0, "end_s": 2.0, "speaker": "A"},
        {"start_s": 2.0, "end_s": 4.0, "speaker": "A"},
        {"start_s": 4.0, "end_s": 6.0, "speaker": "B"},
    ]
    assert tx.speaker_turns(segs) == [4.0]  # one turn, A→B at 4.0
