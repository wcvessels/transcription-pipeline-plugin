"""Timestamp formatting. Replaces v1 transcribe.py:fmt_ts (which could emit ':60')."""


def fmt_ts(seconds, with_hours=True):
    """SRT-style HH:MM:SS,mmm (or MM:SS,mmm). Floors whole seconds, rounds milliseconds; never carries to :60."""
    s = max(0.0, float(seconds))
    total = int(s)                       # whole seconds, floored
    ms = int(round((s - total) * 1000))  # millisecond remainder
    if ms == 1000:                       # guard the rounding edge (e.g. 1.9996)
        total += 1
        ms = 0
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if with_hours:
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    return f"{m:02d}:{sec:02d},{ms:03d}"


def fmt_clock(seconds):
    """Compact HHMMSS for filenames/anchors. Truncates; no rounding carry."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}{m:02d}{sec:02d}"
