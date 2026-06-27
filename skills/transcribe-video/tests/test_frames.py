from pathlib import Path

import imagehash
import numpy as np
import pytest
from PIL import Image

import frames


def _h(bits):
    """Build an ImageHash from explicit bits so consecutive Hamming distances are exact in tests."""
    return imagehash.ImageHash(np.array(bits, dtype=bool))

FIXTURE = Path(__file__).parent / "fixtures" / "scenes_showinfo.txt"


# ---- content-box auto-trim (dense change-detection; excludes constant borders) ----

def _bordered(center_val, size=100, border=10):
    """A frame with a CONSTANT gray border and a center region set to center_val."""
    a = np.full((size, size), 128, dtype="uint8")
    a[border:size - border, border:size - border] = center_val
    return Image.fromarray(a, mode="L")


def test_content_box_excludes_constant_border():
    # center varies across frames (0..255), border is constant -> box hugs the varying center
    imgs = [_bordered(0), _bordered(255), _bordered(128)]
    left, top, right, bottom = frames.content_box(imgs, var_threshold=12)
    assert left == pytest.approx(0.10, abs=0.02)
    assert top == pytest.approx(0.10, abs=0.02)
    assert right == pytest.approx(0.90, abs=0.02)
    assert bottom == pytest.approx(0.90, abs=0.02)


def test_content_box_all_constant_returns_full_frame():
    # nothing varies -> no signal to localize -> fall back to the whole frame (early-return guard)
    assert frames.content_box([_bordered(128), _bordered(128)], var_threshold=12) == (0.0, 0.0, 1.0, 1.0)


def test_content_box_full_bleed_returns_full_frame():
    # full-bleed recording: variation reaches every edge -> the COMPUTATION yields the whole frame (the
    # deliberate no-op the rewrite relies on). Distinct from the nothing-varies early return above.
    a = np.zeros((10, 10), dtype="uint8")
    b = a.copy()
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = 255   # change touches all four edges -> every row & col active
    box = frames.content_box([Image.fromarray(a, "L"), Image.fromarray(b, "L")], var_threshold=12)
    assert box == (0.0, 0.0, 1.0, 1.0)


def test_crop_to_box_fractions_to_pixels():
    out = frames.crop_to_box(Image.new("L", (100, 200)), (0.1, 0.2, 0.9, 0.8))
    assert out.size == (80, 120)   # (0.9-0.1)*100, (0.8-0.2)*200


def test_crop_to_box_full_frame_is_identity_size():
    assert frames.crop_to_box(Image.new("RGB", (64, 48)), (0.0, 0.0, 1.0, 1.0)).size == (64, 48)


# ---- scene-cut proximity (alignment-anchor flag, tolerant of sub-second cuts vs integer-second frames) ----

def test_frame_at_cut_flags_nearest_frame_within_tolerance():
    # ffmpeg cut at 9.408; at 1 fps the nearest sampled frame is 9.0 (0.408 <= 0.5 tolerance)
    assert frames.frame_at_cut(9.0, [9.408, 31.4], tolerance=0.5) is True
    assert frames.frame_at_cut(10.0, [9.408, 31.4], tolerance=0.5) is False   # 0.592 > 0.5
    assert frames.frame_at_cut(5.0, [], tolerance=0.5) is False


# ---- scene change-detection (segment dense frames into held scenes) ----

def test_segment_scenes_held_run_is_one_segment():
    h = _h([0, 0, 0, 0])
    assert frames.segment_scenes([h, h, h], threshold=2) == [[0, 1, 2]]


def test_segment_scenes_delta_equal_threshold_stays_one_segment():
    # boundary contract is strict '>': delta == threshold does NOT open a new scene (guards a >= regression)
    a, b = _h([0, 0, 0, 0]), _h([1, 1, 0, 0])   # Hamming delta == 2
    assert frames.segment_scenes([a, b], threshold=2) == [[0, 1]]


def test_segment_scenes_splits_on_jump():
    a, b = _h([0, 0, 0, 0]), _h([1, 1, 1, 0])   # a - b == 3 > 2
    assert frames.segment_scenes([a, a, b, b], threshold=2) == [[0, 1], [2, 3]]


def test_segment_scenes_small_jitter_stays_one_segment():
    a, c = _h([0, 0, 0, 0]), _h([1, 0, 0, 0])   # delta 1 <= 2 (held-frame jitter)
    assert frames.segment_scenes([a, c, a], threshold=2) == [[0, 1, 2]]


def test_segment_scenes_empty():
    assert frames.segment_scenes([], threshold=2) == []


# ---- scene-start picker: FIRST non-junk per segment (not sharpest) ----

def _rec2(ts, sharp, info):
    return {"timestamp_s": ts, "sharpness": sharp, "info": info}


def test_first_non_junk_per_segment_picks_scene_start_not_sharpest():
    # idx0 junk (blurry); idx1 and idx2 clean. Must pick idx1 (FIRST clean = scene-start),
    # NOT idx2 (sharpest, mid/late hold) — the alignment fix.
    recs = [_rec2(0.0, 5.0, 5.0), _rec2(1.0, 80.0, 5.0), _rec2(2.0, 90.0, 5.0)]
    out = frames.first_non_junk_per_segment(recs, [[0, 1, 2]], blur_floor=10.0, low_info_floor=1.0)
    assert [r["timestamp_s"] for r in out] == [1.0]


def test_first_non_junk_per_segment_skips_leading_junk_within_segment():
    recs = [_rec2(0.0, 1.0, 5.0), _rec2(1.0, 1.0, 5.0), _rec2(2.0, 50.0, 5.0)]  # 0,1 junk; 2 clean
    out = frames.first_non_junk_per_segment(recs, [[0, 1, 2]], blur_floor=10.0, low_info_floor=1.0)
    assert [r["timestamp_s"] for r in out] == [2.0]


def test_first_non_junk_per_segment_drops_all_junk_segment():
    recs = [_rec2(0.0, 1.0, 0.1), _rec2(1.0, 2.0, 0.1), _rec2(2.0, 50.0, 5.0)]
    out = frames.first_non_junk_per_segment(recs, [[0, 1], [2]], blur_floor=10.0, low_info_floor=1.0)
    assert [r["timestamp_s"] for r in out] == [2.0]


# ---- scene-cut parsing (hardened, §10 #3) ----

def test_parse_scene_times_from_showinfo_stderr():
    times = frames.parse_scene_times(FIXTURE.read_text(encoding="utf-8"))
    assert times == [0.0, 3.2, 9.408, 15.018]


def test_parse_scene_times_tolerates_garbage():
    assert frames.parse_scene_times("") == []
    assert frames.parse_scene_times("ffmpeg version 8.1\nno pts here") == []


# ---- scoring + junk filter (real images) ----

def _checker(path):
    a = np.zeros((64, 64), dtype="uint8")
    a[::2, ::2] = 255
    a[1::2, 1::2] = 255
    Image.fromarray(a, mode="L").save(path)


def _flat(path, v=128):
    Image.fromarray(np.full((64, 64), v, dtype="uint8"), mode="L").save(path)


def test_laplacian_variance_sharper_scores_higher(tmp_path):
    sharp = tmp_path / "sharp.png"; _checker(sharp)
    flat = tmp_path / "flat.png"; _flat(flat)
    assert frames.laplacian_variance(Image.open(sharp)) > frames.laplacian_variance(Image.open(flat))


def test_info_entropy_uniform_is_low(tmp_path):
    flat = tmp_path / "flat.png"; _flat(flat)
    checker = tmp_path / "checker.png"; _checker(checker)
    assert frames.info_entropy(Image.open(flat)) < frames.info_entropy(Image.open(checker))


def test_is_junk_flags_blur_and_low_info():
    assert frames.is_junk(sharpness=1.0, info=5.0, blur_floor=10.0, low_info_floor=1.0) is True   # blurry
    assert frames.is_junk(sharpness=50.0, info=0.2, blur_floor=10.0, low_info_floor=1.0) is True  # near-blank
    assert frames.is_junk(sharpness=50.0, info=5.0, blur_floor=10.0, low_info_floor=1.0) is False


# ---- duration-proportional frame cap ----

def test_frame_cap_scales_with_duration():
    assert frames.frame_cap(duration_s=600.0, explicit=None, per_min=40) == 400   # 10 min × 40
    assert frames.frame_cap(duration_s=90.0, explicit=None, per_min=40) == 60     # 1.5 min × 40
    assert frames.frame_cap(duration_s=1.0, explicit=None, per_min=40) == 1       # floor at 1


def test_frame_cap_explicit_overrides_duration():
    assert frames.frame_cap(duration_s=600.0, explicit=50, per_min=40) == 50


def test_frame_cap_then_decimate_bites_at_short_duration():
    # 30s clip, 40/min default -> cap 20; 50 survivors decimate down to the duration-scaled cap
    survivors = [_rec(f"f{i}.jpg", float(i), 50.0) for i in range(50)]
    cap = frames.frame_cap(duration_s=30.0, explicit=None, per_min=40)
    assert cap == 20
    assert len(frames.decimate(survivors, cap)) == 20


def test_explicit_max_frames_beats_duration_cap_end_to_end():
    survivors = [_rec(f"f{i}.jpg", float(i), 50.0) for i in range(50)]
    cap = frames.frame_cap(duration_s=30.0, explicit=10, per_min=40)   # explicit 10 < duration's 20
    assert len(frames.decimate(survivors, cap)) == 10


# ---- decimate (over-capture cap) ----

def _rec(path, ts, sharp):
    return {"file": str(path), "timestamp_s": ts, "sharpness": sharp, "info": 5.0}


def test_decimate_respects_max_frames():
    recs = [_rec(f"f{i}.jpg", float(i), 50.0) for i in range(250)]
    out = frames.decimate(recs, max_frames=100)
    assert len(out) == 100
    assert out[0]["timestamp_s"] == 0.0
