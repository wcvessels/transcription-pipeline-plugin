"""Frame curation: scene-detect (hardened) + best-of-window selection + phash dedup. §4.4.

Best-of-window (locked decision #6): a settle window opens at each scene cut (or a fixed cadence
when no cuts are found); we sample `window_size` candidates across it, score each for sharpness
(variance of a numpy Laplacian) and information (histogram entropy), drop junk (blurry / near-blank),
keep the SHARPEST non-junk candidate, then perceptual-hash-dedup the survivors. window_size=1
collapses to plain extract-then-dedup (the escape hatch). Scene-CUT timestamps are returned as a
distinct artifact from the curated frames (locked decision #3): alignment anchors are cut times,
not the displayed frame's time, so moving the best-of-window pick never perturbs alignment.

Pure functions (select_windows, candidate_timestamps, laplacian_variance, info_entropy, is_junk,
best_of_window, decimate) are unit-tested directly. detect_scenes/extract_frame shell out to ffmpeg
and run only in the orchestrator + the gated e2e test."""
import re
import subprocess

import imagehash
import numpy as np
from PIL import Image

_PTS_RE = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")


# ---- scene-cut parsing (hardened, §10 #3) ----

def parse_scene_times(stderr_text: str) -> list:
    """Extract scene-change pts_time values from ffmpeg showinfo stderr.
    Hardened (§10 #3): returns [] rather than raising on unexpected output."""
    out = []
    try:
        for m in _PTS_RE.finditer(stderr_text or ""):
            out.append(float(m.group(1)))
    except (ValueError, TypeError):
        return []
    return out


def detect_scenes(video_path, threshold: float) -> list:
    """Run ffmpeg scene filter; return scene-cut timestamps (seconds)."""
    proc = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-filter:v",
         f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return parse_scene_times(proc.stderr)


# ---- settle-window selection + candidate sampling (pure) ----

def select_windows(duration_s, scene_times, interval_seconds, frames_per_minute, settle_s=1.0):
    """Return [(start, end)] settle windows. Priority: explicit interval > scene cuts > fixed cadence.
    Each window opens at a start time and spans settle_s OR until the next start, whichever is sooner,
    so a cut quickly followed by another cut gets a short window, never an overlap. Never empty."""
    if interval_seconds:
        n = int(duration_s // interval_seconds)
        starts = [i * interval_seconds for i in range(n + 1)]
    elif scene_times:
        starts = sorted({t for t in scene_times if 0.0 <= t <= duration_s})
        if not starts or starts[0] > 0.0:
            starts = [0.0] + starts
    else:
        fpm = frames_per_minute or 5
        step = max(1.0, 60.0 / fpm)
        n = int(duration_s // step)
        starts = [i * step for i in range(max(1, n + 1))]
    windows = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else duration_s
        end = min(s + settle_s, nxt, duration_s)
        if end <= s:
            end = min(s + 0.001, duration_s)
        windows.append((float(s), float(end)))
    return windows


def candidate_timestamps(window, window_size):
    """Evenly spaced candidate times inside a window. window_size<=1 → the single midpoint."""
    a, b = window
    if window_size <= 1 or b <= a:
        return [(a + b) / 2.0]
    return [a + i * (b - a) / (window_size - 1) for i in range(window_size)]


# ---- scoring + junk filter (real images) ----

def laplacian_variance(img):
    """Sharpness proxy: variance of a discrete Laplacian over grayscale (numpy, no cv2/scipy)."""
    g = np.asarray(img.convert("L"), dtype=np.float64)
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var()) if lap.size else 0.0


def info_entropy(img):
    """Low-info proxy: Shannon entropy of the grayscale histogram (Pillow). Near-blank → low."""
    return float(img.convert("L").entropy())


def is_junk(sharpness, info, blur_floor, low_info_floor):
    """A candidate is junk if it is too blurry OR carries too little information (near-blank)."""
    return sharpness < blur_floor or info < low_info_floor


def score_frame(path):
    """Score one extracted candidate. Returns {sharpness, info}."""
    img = Image.open(path)
    return {"sharpness": laplacian_variance(img), "info": info_entropy(img)}


def best_of_window(scored, blur_floor, low_info_floor):
    """scored = [{file, timestamp_s, sharpness, info}]. Return (best_non_junk_or_None, n_junk).
    Best = highest sharpness among non-junk candidates; None when the whole window is junk."""
    non_junk = [s for s in scored if not is_junk(s["sharpness"], s["info"], blur_floor, low_info_floor)]
    n_junk = len(scored) - len(non_junk)
    if not non_junk:
        return None, n_junk
    return max(non_junk, key=lambda s: s["sharpness"]), n_junk


# ---- extraction + dedup ----

def extract_frame(video_path, timestamp, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _hashes(img_path):
    """Compute BOTH perceptual hashes for one image (Will decision 2026-06-15, phash-vs-colorhash
    informed comparison). phash = DCT structural hash — the §4.4 contract that drives LIVE dedup;
    colorhash = HSV color-distribution hash, computed alongside for the comparison only."""
    img = Image.open(img_path)
    return imagehash.phash(img), imagehash.colorhash(img)


def phash_dedup(records, threshold: int, method: str = "phash"):
    """Drop a survivor if its hash is within `threshold` Hamming distance of the previous KEPT
    survivor. threshold=0 disables. LIVE dedup is driven by `method` (default 'phash', the §4.4
    contract). BOTH hashes are computed and carried on each kept record: `phash` is the §16.4
    manifest field; `colorhash` is an in-memory diagnostic for the phash-vs-colorhash comparison
    (the orchestrator does NOT write it to the manifest). Returns (kept_records, dropped_count);
    kept records get a 0..k-1 index plus `phash` and `colorhash` hex strings."""
    kept = []
    last = None
    dropped = 0
    for rec in records:
        ph, ch = _hashes(rec["file"])
        active = ph if method == "phash" else ch
        if threshold > 0 and last is not None and (active - last) <= threshold:
            dropped += 1
            continue  # near-duplicate of the last kept survivor (by the active method)
        rec = dict(rec)
        rec["phash"] = str(ph)
        rec["colorhash"] = str(ch)
        kept.append(rec)
        last = active
    for i, rec in enumerate(kept):
        rec["index"] = i
    return kept, dropped


def compare_dedup_methods(records, threshold: int) -> dict:
    """Run dedup under BOTH phash and colorhash over the SAME input set, for an informed
    phash-vs-colorhash decision (Will 2026-06-15). Returns {'phash': {...}, 'colorhash': {...}},
    each with kept_count, dropped, dedup_reduction (dropped / count). Pure comparison: does not
    mutate `records` and does not pick a winner — the live pipeline dedups by phash."""
    n = len(records)
    out = {}
    for m in ("phash", "colorhash"):
        kept, dropped = phash_dedup(records, threshold, method=m)
        out[m] = {
            "kept_count": len(kept),
            "dropped": dropped,
            "dedup_reduction": round(dropped / n, 4) if n else 0.0,
        }
    return out


def decimate(records, max_frames: int) -> list:
    """Evenly thin survivor records down to max_frames (§4.4 hard cap), preserving order."""
    if len(records) <= max_frames:
        return list(records)
    step = len(records) / max_frames
    return [records[int(i * step)] for i in range(max_frames)]
