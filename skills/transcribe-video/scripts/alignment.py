"""Joint-signal alignment (§4.5, local mode). Replaces v1's magic-number window.

Anchors = sorted union of scene cuts + speaker turns + silence-gap midpoints +
caption-cue boundaries. Frames partition the timeline; each whole segment lands
under one frame by its midpoint. No segment splitting (locked decision #4)."""


def build_anchors(scene_cuts, speaker_turns, silence_gaps, caption_cues, duration, merge_eps=0.25):
    raw = [0.0, float(duration)]
    for group in (scene_cuts, speaker_turns, silence_gaps, caption_cues):
        raw.extend(float(t) for t in group)
    raw = sorted(t for t in raw if 0.0 <= t <= duration)
    merged = []
    for t in raw:
        if not merged or t - merged[-1] > merge_eps:
            merged.append(t)
    if merged[0] > 0.0:
        merged.insert(0, 0.0)
    if merged[-1] < duration:
        merged.append(float(duration))
    return merged


def _boundary(x_ts, y_ts, anchors):
    """Ownership boundary between consecutive frames X, Y per the pinned rule."""
    midpoint = (x_ts + y_ts) / 2.0
    between = [a for a in anchors if x_ts < a < y_ts]
    if not between:               # same cell → split at midpoint
        return midpoint
    # nearest anchor to the midpoint; ties → lower anchor (sorted, stable)
    return min(between, key=lambda a: (abs(a - midpoint), a))


def frame_intervals(frames, anchors, duration):
    """Partition [0, duration] into one owned interval per frame. Returns [[index, start, end], ...]."""
    fr = sorted(frames, key=lambda f: f["timestamp_s"])
    if not fr:
        return []
    if len(fr) == 1:
        return [[fr[0]["index"], 0.0, float(duration)]]
    bounds = [0.0]
    for x, y in zip(fr, fr[1:]):
        bounds.append(_boundary(x["timestamp_s"], y["timestamp_s"], anchors))
    bounds.append(float(duration))
    return [[fr[k]["index"], bounds[k], bounds[k + 1]] for k in range(len(fr))]


def align(frames, segments, anchors, duration):
    """Assign each segment a frame_index by its midpoint. Mutates+returns the segment list."""
    intervals = frame_intervals(frames, anchors, duration)
    if not intervals:
        for seg in segments:
            seg["frame_index"] = None
        return segments
    for seg in segments:
        mid = (float(seg["start_s"]) + float(seg["end_s"])) / 2.0
        chosen = intervals[-1][0]  # default to last frame (covers mid == duration)
        for idx, start, end in intervals:
            if start <= mid < end:
                chosen = idx
                break
        seg["frame_index"] = chosen
    return segments
