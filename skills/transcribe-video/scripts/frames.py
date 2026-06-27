"""Frame curation: dense sampling + perceptual-hash change-detection. §4.4.

Dense change-detection (supersedes best-of-window): extract frames at a fixed cadence in one ffmpeg
pass, trim any constant letterbox/pillarbox borders (content_box), score sharpness + info and drop junk,
then segment the timeline into held SCENES by the phash delta between CONSECUTIVE frames (segment_scenes),
and keep the FIRST non-junk frame of each scene (first_non_junk_per_segment) — its timestamp marks when
the screen appeared, the alignment key. One change signal replaces both the old ffmpeg pixel-delta
SAMPLING (too weak — only fired at hard cuts, so smooth-transition bodies got no windows) and the old
absolute-similarity dedup (collapsed distinct screens sharing app chrome). Completeness is set by the
change threshold (low = more captures); held screens don't over-segment because they're static at hash
resolution. Scene-CUT timestamps from ffmpeg are still returned, but only as alignment anchors
(decision #3), decoupled from sampling.

Pure functions (content_box, segment_scenes, first_non_junk_per_segment, laplacian_variance,
info_entropy, is_junk, decimate) are unit-tested directly. detect_scenes/extract_frames_fps shell out
to ffmpeg and run only in the orchestrator + the gated e2e test."""
import math
import re
import subprocess
from pathlib import Path

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


def frame_at_cut(ts, scene_times, tolerance) -> bool:
    """True if any ffmpeg scene cut falls within `tolerance` seconds of this frame's timestamp — i.e. this
    sampled frame is the one nearest that cut (the manifest `is_scene_cut` alignment-anchor flag). Replaces
    brittle 0.1s-bucket equality, which almost never matched at 1 fps (frame ts are integer seconds but
    cuts are arbitrary sub-second floats). Pass tolerance = half the sample period."""
    return any(abs(ts - c) <= tolerance for c in scene_times)


# ---- content-box auto-trim (dense change-detection) ----

def content_box(images, var_threshold=12):
    """Bounding box (left/top/right/bottom FRACTIONS) of the pixels that VARY across `images` by more
    than var_threshold — i.e. trims the GLOBALLY-CONSTANT outer borders (letterbox/pillarbox bars). On a
    full-bleed screen recording nothing is globally constant (chrome, taskbar and webcam all change at
    some point), so it returns the full frame — a deliberate no-op there; its job is letterboxed sources.
    Falls back to the full frame when nothing varies. `images` are same-size PIL frames across the clip."""
    stack = np.stack([np.asarray(im.convert("L"), dtype=np.int16) for im in images])
    rng = stack.max(axis=0) - stack.min(axis=0)
    active = rng > var_threshold
    if not active.any():
        return (0.0, 0.0, 1.0, 1.0)
    h, w = active.shape
    rows = np.where(active.any(axis=1))[0]
    cols = np.where(active.any(axis=0))[0]
    return (float(cols[0] / w), float(rows[0] / h), float((cols[-1] + 1) / w), float((rows[-1] + 1) / h))


def crop_to_box(img, box):
    """Crop a PIL image to a fractional (left, top, right, bottom) content box from content_box()."""
    w, h = img.size
    left, top, right, bottom = box
    return img.crop((int(left * w), int(top * h), int(right * w), int(bottom * h)))


def content_box_from_paths(paths, var_threshold=12):
    """content_box over frames given by path — orchestrator seam (keeps PIL out of the orchestrator).
    Loads each frame inside a `with` so file handles are released before the work dir is rmtree'd
    (Windows holds the handle until GC otherwise)."""
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("L"))   # convert() reads the pixels; the handle closes with the block
    return content_box(imgs, var_threshold)


def score_and_hash(path, box, hash_size=16):
    """Per dense frame: sharpness + info (for the junk filter) AND the change hash that segment_scenes
    consumes (computed on the content box). Returns {sharpness, info, hash}. The frame is opened in a
    `with` so its file handle is released before the work dir is cleaned up (Windows)."""
    with Image.open(path) as img:
        return {"sharpness": laplacian_variance(img), "info": info_entropy(img),
                "hash": imagehash.phash(crop_to_box(img, box), hash_size=hash_size)}


# ---- scene change-detection (segment dense frames into held scenes) ----

def segment_scenes(hashes, threshold):
    """Group ordered per-frame content-region hashes into held-scene segments: a new segment opens
    when the perceptual-hash delta from the previous frame exceeds `threshold` (a screen change). Returns
    a list of segments, each a list of frame indices; held frames (small jitter) stay together. This is
    the single mechanism that replaces BOTH ffmpeg pixel-delta scene detection (too weak — diluted by
    fixed chrome) and absolute-similarity dedup (collapsed distinct screens sharing app chrome): it keys
    on the TRANSITION between screens, not their absolute similarity, so distinct-but-similar screens are
    kept while a held screen collapses to one segment."""
    if not hashes:
        return []
    segments = [[0]]
    for i in range(1, len(hashes)):
        if (hashes[i] - hashes[i - 1]) > threshold:
            segments.append([i])
        else:
            segments[-1].append(i)
    return segments


def first_non_junk_per_segment(records, segments, blur_floor, low_info_floor):
    """Per scene segment, return the FIRST non-junk frame (scene-START) — NOT the sharpest. The first
    clear frame marks when the screen became visible, which is the alignment key: a transcript line
    "this is the X screen" at time t must align to the frame stamped t, so A2 can map narration to the
    screen on display. (Sharpest-per-segment stamped held scenes at the END of the hold — the 48/49
    bug.) Segments whose frames are all junk are dropped. `records[i]` carries sharpness + info."""
    out = []
    for seg in segments:
        for i in seg:
            r = records[i]
            if not is_junk(r["sharpness"], r["info"], blur_floor, low_info_floor):
                out.append(r)
                break
    return out


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


# ---- dense extraction + decimation ----

def extract_frames_fps(video_path, rate, out_dir):
    """Dense sampling: ONE ffmpeg decode pass emits `rate` frames/sec as d_000001.jpg…. Returns
    [(path, timestamp_s)] in order (frame i at i/rate s). One pass replaces the old per-candidate
    -ss seeks (one decode, fewer process spawns). Shells ffmpeg — exercised in the gated e2e test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
         "-vf", f"fps={rate}", "-q:v", "2", str(out_dir / "d_%06d.jpg")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # sort by the numeric frame index, not lexicographically — d_1000000.jpg must follow d_999999.jpg
    files = sorted(out_dir.glob("d_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    return [(p, i / float(rate)) for i, p in enumerate(files)]


def frame_cap(duration_s, explicit, per_min):
    """Max kept frames. An explicit --max-frames wins; otherwise scale by duration (per_min per minute)
    so long sessions are never capped by a flat number — only pathological full-motion video hits it.
    Floor of 1."""
    if explicit is not None:
        return explicit
    return max(1, math.ceil(per_min * duration_s / 60.0))


def decimate(records, max_frames: int) -> list:
    """Evenly thin survivor records down to max_frames (§4.4 hard cap), preserving order."""
    if len(records) <= max_frames:
        return list(records)
    step = len(records) / max_frames
    return [records[int(i * step)] for i in range(max_frames)]
