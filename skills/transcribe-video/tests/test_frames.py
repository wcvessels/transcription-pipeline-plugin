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


def test_first_non_junk_per_segment_image_is_first_clean_but_timestamp_is_scene_start():
    # (B / Gemini #8) the kept frame's IMAGE is the first non-junk frame (not the sharpest), but its
    # alignment timestamp is decoupled to the SCENE START (segment[0]). idx0 is junk (blurry); the image
    # is idx1's (first clean, sharpness 80, NOT idx2's sharper 90), stamped at idx0's time 0.0.
    recs = [_rec2(0.0, 5.0, 5.0), _rec2(1.0, 80.0, 5.0), _rec2(2.0, 90.0, 5.0)]
    out = frames.first_non_junk_per_segment(recs, [[0, 1, 2]], blur_floor=10.0, low_info_floor=1.0)
    assert len(out) == 1
    assert out[0]["sharpness"] == 80.0     # IMAGE = first clean frame (idx1), not the sharpest (idx2)
    assert out[0]["timestamp_s"] == 0.0    # TIMESTAMP = scene start (segment[0]=idx0), not idx1's 1.0


def test_first_non_junk_per_segment_timestamp_is_scene_start_even_skipping_leading_junk():
    # idx0,idx1 junk; idx2 first clean. Image = idx2 (sharpness 50); timestamp = scene start 0.0.
    recs = [_rec2(0.0, 1.0, 5.0), _rec2(1.0, 1.0, 5.0), _rec2(2.0, 50.0, 5.0)]
    out = frames.first_non_junk_per_segment(recs, [[0, 1, 2]], blur_floor=10.0, low_info_floor=1.0)
    assert out[0]["sharpness"] == 50.0
    assert out[0]["timestamp_s"] == 0.0    # NOT idx2's 2.0 — the screen appeared at the scene start


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


def test_decimate_preserves_last_record():
    # (Codex F2 / Gemini #3) when the cap bites, the FINAL scene (the video's ending state) must be kept,
    # not silently truncated. 41 survivors capped to 40 must keep BOTH endpoints (0.0 and 40.0).
    recs = [_rec(f"f{i}.jpg", float(i), 50.0) for i in range(41)]
    out = frames.decimate(recs, max_frames=40)
    assert len(out) == 40
    assert out[0]["timestamp_s"] == 0.0
    assert out[-1]["timestamp_s"] == 40.0   # the tail-truncation bug dropped this


def test_decimate_max_frames_one_keeps_first():
    recs = [_rec(f"f{i}.jpg", float(i), 50.0) for i in range(5)]
    assert [r["timestamp_s"] for r in frames.decimate(recs, max_frames=1)] == [0.0]


def test_decimate_max_frames_zero_returns_empty():
    # (Codex F6 / Gemini #6) defensive: a non-positive cap returns [] instead of ZeroDivisionError.
    recs = [_rec(f"f{i}.jpg", float(i), 50.0) for i in range(5)]
    assert frames.decimate(recs, max_frames=0) == []


# ---- (A) anchor-baseline segmentation: catches accumulated slow drift ----

def test_segment_scenes_splits_on_accumulated_drift():
    # Consecutive deltas are all 1 (<= threshold 2) but cumulative drift from the segment ANCHOR grows.
    # Anchor-baseline must split when drift from the scene's first frame exceeds the threshold, so a slow
    # crossfade between two distinct screens is not collapsed into one segment (Codex C1 / Gemini #5).
    h0 = _h([0, 0, 0, 0, 0, 0])
    h1 = _h([1, 0, 0, 0, 0, 0])   # +1 from h0
    h2 = _h([1, 1, 0, 0, 0, 0])   # +1 from h1, +2 from h0
    h3 = _h([1, 1, 1, 0, 0, 0])   # +1 from h2, +3 from h0 -> breaches threshold 2
    h4 = _h([1, 1, 1, 1, 0, 0])   # +1 from h3
    assert frames.segment_scenes([h0, h1, h2, h3, h4], threshold=2) == [[0, 1, 2], [3, 4]]


def test_segment_scenes_bounded_jitter_stays_one_segment_under_anchor():
    # The tradeoff guard: anchor-baseline must NOT over-split held-screen jitter that oscillates within
    # the threshold of the anchor (never drifting away) — it stays one segment.
    a = _h([0, 0, 0, 0])
    j = _h([1, 0, 0, 0])          # delta 1 from anchor, oscillating back and forth
    assert frames.segment_scenes([a, j, a, j, a], threshold=2) == [[0, 1, 2, 3, 4]]


# ---- content_box over ALL given frames (no subsample blind spot) ----

def test_content_box_from_paths_considers_all_frames(tmp_path):
    # (Codex F1 / Gemini #1) the box must reflect EVERY frame it is given: a region active in only ONE
    # frame still widens the box. (The orchestrator now passes all dense frames, not a 24-sample subset.)
    base = np.full((100, 100), 128, dtype="uint8")   # constant gray everywhere
    paths = []
    for i in range(8):
        a = base.copy()
        if i == 5:                                    # exactly one frame lights up the top-left corner
            a[0:10, 0:10] = 255
        p = tmp_path / f"f{i}.png"; Image.fromarray(a, "L").save(p); paths.append(str(p))
    left, top, right, bottom = frames.content_box_from_paths(paths, var_threshold=12)
    assert left == 0.0 and top == 0.0                 # box includes the corner that varied in one frame


# ---- score_and_hash scores the CONTENT BOX, not the full frame ----

def test_score_and_hash_scores_the_content_box_not_the_full_frame(tmp_path):
    # (Codex F5) sharpness + info must be measured on the cropped content box, not the full frame, or a
    # letterboxed source's black bars dilute a valid frame below the junk floors.
    a = np.zeros((100, 100), dtype="uint8")
    a[40:60, 40:60] = np.tile([0, 255], 200).reshape(20, 20)   # busy 20x20 center, black bars around
    p = tmp_path / "letterboxed.png"; Image.fromarray(a, mode="L").save(p)
    box = (0.4, 0.4, 0.6, 0.6)                                  # isolates the busy center
    s = frames.score_and_hash(str(p), box)
    cropped = frames.crop_to_box(Image.open(p), box)
    assert s["sharpness"] == pytest.approx(frames.laplacian_variance(cropped))
    assert s["info"] == pytest.approx(frames.info_entropy(cropped))
    assert frames.laplacian_variance(Image.open(p)) < s["sharpness"]   # full frame is diluted -> lower


# ---- detect_scenes zero-bases its timestamps to match the frame clock ----

def test_detect_scenes_zero_bases_timestamps(monkeypatch):
    # (Gemini #4) extract_frames_fps assigns synthetic 0-based timestamps; detect_scenes must prepend
    # setpts=PTS-STARTPTS so a container start-time offset cannot silently desync is_scene_cut.
    captured = {}
    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        class _P: stderr = ""
        return _P()
    monkeypatch.setattr(frames.subprocess, "run", _fake_run)
    frames.detect_scenes("video.mp4", 0.3)
    vf = captured["cmd"][captured["cmd"].index("-filter:v") + 1]
    assert vf.startswith("setpts=PTS-STARTPTS,")
    assert "select=" in vf and vf.index("setpts") < vf.index("select")
