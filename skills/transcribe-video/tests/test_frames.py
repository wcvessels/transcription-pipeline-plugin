from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import frames

FIXTURE = Path(__file__).parent / "fixtures" / "scenes_showinfo.txt"


# ---- scene-cut parsing (hardened, §10 #3) ----

def test_parse_scene_times_from_showinfo_stderr():
    times = frames.parse_scene_times(FIXTURE.read_text(encoding="utf-8"))
    assert times == [0.0, 3.2, 9.408, 15.018]


def test_parse_scene_times_tolerates_garbage():
    assert frames.parse_scene_times("") == []
    assert frames.parse_scene_times("ffmpeg version 8.1\nno pts here") == []


# ---- settle-window selection (pure) ----

def test_select_windows_from_scene_cuts():
    # a settle window opens at each cut, capped at settle_s OR the next cut, whichever is sooner
    w = frames.select_windows(duration_s=12.0, scene_times=[0.0, 5.0, 10.0],
                              interval_seconds=None, frames_per_minute=5, settle_s=1.0)
    assert w == [(0.0, 1.0), (5.0, 6.0), (10.0, 11.0)]


def test_select_windows_caps_at_next_cut():
    w = frames.select_windows(duration_s=12.0, scene_times=[0.0, 0.5],
                              interval_seconds=None, frames_per_minute=5, settle_s=1.0)
    assert w[0] == (0.0, 0.5)            # capped by the next cut, not settle_s
    assert w[1][0] == 0.5


def test_select_windows_no_scenes_fixed_cadence():
    # no cuts + no interval → fixed cadence at frames_per_minute, never empty (§4.4 edge case)
    w = frames.select_windows(duration_s=120.0, scene_times=[], interval_seconds=None,
                              frames_per_minute=5, settle_s=1.0)
    assert len(w) >= 1


def test_select_windows_interval_override():
    w = frames.select_windows(duration_s=30.0, scene_times=[1.0, 2.0], interval_seconds=5.0,
                              frames_per_minute=5, settle_s=1.0)
    assert [round(a, 1) for a, _ in w] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]


# ---- candidate sampling within a window (pure) ----

def test_candidate_timestamps_evenly_spaced():
    assert frames.candidate_timestamps((0.0, 1.0), window_size=5) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_candidate_timestamps_window_size_one_is_midpoint():
    # window_size=1 is the escape hatch: one candidate at the window midpoint
    assert frames.candidate_timestamps((4.0, 6.0), window_size=1) == [5.0]


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


def test_best_of_window_picks_sharpest_non_junk():
    scored = [
        {"file": "a.jpg", "timestamp_s": 0.0, "sharpness": 5.0, "info": 5.0},    # junk: blurry
        {"file": "b.jpg", "timestamp_s": 0.5, "sharpness": 80.0, "info": 5.0},   # best
        {"file": "c.jpg", "timestamp_s": 1.0, "sharpness": 40.0, "info": 5.0},
    ]
    best, n_junk = frames.best_of_window(scored, blur_floor=10.0, low_info_floor=1.0)
    assert best["file"] == "b.jpg" and n_junk == 1


def test_best_of_window_all_junk_returns_none():
    scored = [{"file": "a.jpg", "timestamp_s": 0.0, "sharpness": 1.0, "info": 0.1}]
    best, n_junk = frames.best_of_window(scored, blur_floor=10.0, low_info_floor=1.0)
    assert best is None and n_junk == 1


# ---- phash dedup over survivor records (carries timestamp_s + sharpness) ----

def _solid_rgb(path, color):
    Image.new("RGB", (64, 64), color).save(path)


def _checker_rgb(path, block=8):
    a = np.zeros((64, 64, 3), dtype="uint8")
    for r in range(0, 64, block):
        for c in range(0, 64, block):
            if ((r // block) + (c // block)) % 2 == 0:
                a[r:r + block, c:c + block] = 255
    Image.fromarray(a, mode="RGB").save(path)


def _rec(path, ts, sharp):
    return {"file": str(path), "timestamp_s": ts, "sharpness": sharp, "info": 5.0}


def test_phash_dedup_drops_identical_keeps_distinct(tmp_path):
    # phash keys on STRUCTURE: three identical solid reds collapse to one; a structurally distinct
    # frame (checkerboard) survives. (Two solid colors alone are indistinguishable to phash by
    # design; colorhash is computed alongside for the phash-vs-colorhash comparison, Will 2026-06-15.)
    recs = []
    for i in range(3):
        p = tmp_path / f"frame_{i:04d}.jpg"; _solid_rgb(p, (255, 0, 0))
        recs.append(_rec(p, float(i), 50.0 + i))
    chk = tmp_path / "frame_0003.jpg"; _checker_rgb(chk)
    recs.append(_rec(chk, 3.0, 53.0))
    kept, dropped = frames.phash_dedup(recs, threshold=5)
    assert len(kept) == 2 and dropped == 2   # three identical reds → one; checkerboard survives
    assert kept[0]["index"] == 0
    assert all("phash" in k for k in kept)
    assert kept[0]["sharpness"] == 50.0      # sharpness carried through dedup


def test_phash_dedup_threshold_zero_disables(tmp_path):
    recs = []
    for i in range(3):
        p = tmp_path / f"frame_{i:04d}.jpg"; _solid_rgb(p, (255, 0, 0))
        recs.append(_rec(p, float(i), 50.0))
    kept, dropped = frames.phash_dedup(recs, threshold=0)
    assert len(kept) == 3 and dropped == 0


def test_dedup_carries_both_hashes(tmp_path):
    # Will decision: both phash AND colorhash are computed and carried on each kept record.
    recs = []
    for i, color in enumerate([(255, 0, 0), (0, 128, 255)]):
        p = tmp_path / f"frame_{i:04d}.jpg"; _solid_rgb(p, color)
        recs.append(_rec(p, float(i), 50.0))
    kept, _ = frames.phash_dedup(recs, threshold=0)  # threshold 0 → keep both, just check carry
    assert all("phash" in k and "colorhash" in k for k in kept)


def test_compare_dedup_methods_reports_both(tmp_path):
    # The informed decision point: report what BOTH methods would do over the same survivor set.
    recs = []
    for i in range(3):
        p = tmp_path / f"frame_{i:04d}.jpg"; _solid_rgb(p, (255, 0, 0))
        recs.append(_rec(p, float(i), 50.0))
    chk = tmp_path / "frame_0003.jpg"; _checker_rgb(chk)
    recs.append(_rec(chk, 3.0, 50.0))
    cmp = frames.compare_dedup_methods(recs, threshold=5)
    assert set(cmp) == {"phash", "colorhash"}
    for m in ("phash", "colorhash"):
        assert {"kept_count", "dropped", "dedup_reduction"} <= set(cmp[m])
    # phash collapses the 3 uniform reds and keeps the structural checkerboard → 2 kept
    assert cmp["phash"]["kept_count"] == 2


def test_synthetic_dedup_floor_drops_duplicates_keeps_distinct(tmp_path):
    """Synthetic dedup-correctness gate (round-9 recalibration, 2026-06-18) — the controlled-fixture
    home of the property the §16.7 #6 corpus floor used to (mis-)check. On input with KNOWN redundancy,
    phash dedup must shed the duplicates WITHOUT dropping perceptually-distinct frames. The real-corpus
    `dedup_reduction >= 0.30` floor was dropped (two real solo clips legitimately measured 0.14 and 0.00)
    because best-of-window removes redundancy UPSTREAM, so a well-curated real clip yields mostly-distinct
    survivors — uncontrolled redundancy is the wrong thing to gate. Here redundancy IS controlled."""
    recs = []
    for i in range(8):                                    # 8 byte-identical frames (uniform → identical phash)
        p = tmp_path / f"frame_{i:04d}.jpg"; _solid_rgb(p, (200, 50, 50))
        recs.append(_rec(p, float(i), 50.0))
    chk = tmp_path / "frame_0008.jpg"; _checker_rgb(chk)  # 1 structurally-distinct frame that MUST survive
    recs.append(_rec(chk, 8.0, 50.0))
    kept, dropped = frames.phash_dedup(recs, threshold=5)
    assert dropped == 7                                   # the 7 trailing duplicates collapse into the first
    assert dropped / len(recs) >= 0.30                    # sheds a meaningful fraction (the floor's quantitative intent)
    assert len(kept) == 2                                 # one representative duplicate + the distinct frame
    assert len({k["phash"] for k in kept}) == 2           # survivors are perceptually distinct (no over-merge)


def test_decimate_respects_max_frames():
    recs = [_rec(f"f{i}.jpg", float(i), 50.0) for i in range(250)]
    out = frames.decimate(recs, max_frames=100)
    assert len(out) == 100
    assert out[0]["timestamp_s"] == 0.0
