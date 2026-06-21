import alignment


def _frames(ts_list):
    return [{"index": i, "timestamp_s": t, "file": f"f{i}.jpg", "is_scene_cut": True, "phash": "0"}
            for i, t in enumerate(ts_list)]


def test_build_anchors_union_and_merge():
    a = alignment.build_anchors(
        scene_cuts=[3.0, 3.1], speaker_turns=[], silence_gaps=[6.0],
        caption_cues=[0.0, 9.0], duration=10.0, merge_eps=0.25,
    )
    # 3.0 and 3.1 merge; 0 and 10 bookend
    assert a[0] == 0.0 and a[-1] == 10.0
    assert 3.0 in a and 3.1 not in a
    assert 6.0 in a and 9.0 in a


def test_frame_intervals_total_coverage():
    fr = _frames([1.0, 5.0, 9.0])
    anchors = alignment.build_anchors([], [], [], [], duration=10.0)
    intervals = alignment.frame_intervals(fr, anchors, duration=10.0)
    assert intervals[0][1] == 0.0           # first starts at 0
    assert intervals[-1][2] == 10.0         # last ends at duration
    # contiguous
    for (i0), (i1) in zip(intervals, intervals[1:]):
        assert i0[2] == i1[1]


def test_two_frames_one_cell_split_at_midpoint():
    # no anchors between 2.0 and 4.0 → boundary at midpoint 3.0
    fr = _frames([2.0, 4.0])
    anchors = alignment.build_anchors([], [], [], [], duration=6.0)
    intervals = alignment.frame_intervals(fr, anchors, duration=6.0)
    assert intervals[0] == [0, 0.0, 3.0]
    assert intervals[1] == [1, 3.0, 6.0]


def test_empty_cell_annexes_to_nearer_frame_via_anchor():
    # frames at 1 and 9; an anchor at 4 sits between them; midpoint is 5 → nearest anchor 4
    fr = _frames([1.0, 9.0])
    anchors = alignment.build_anchors(scene_cuts=[4.0], speaker_turns=[], silence_gaps=[],
                                      caption_cues=[], duration=10.0)
    intervals = alignment.frame_intervals(fr, anchors, duration=10.0)
    assert intervals[0] == [0, 0.0, 4.0]
    assert intervals[1] == [1, 4.0, 10.0]


def test_segment_assigned_by_midpoint_no_split():
    fr = _frames([1.0, 9.0])
    anchors = alignment.build_anchors([5.0], [], [], [], duration=10.0)
    segs = [
        {"index": 0, "start_s": 0.0, "end_s": 2.0, "speaker": None, "text": "a", "frame_index": None},
        {"index": 1, "start_s": 6.0, "end_s": 8.0, "speaker": None, "text": "b", "frame_index": None},
        {"index": 2, "start_s": 3.0, "end_s": 9.0, "speaker": None, "text": "long", "frame_index": None},  # mid=6 → frame 1
    ]
    out = alignment.align(fr, segs, anchors, duration=10.0)
    assert out[0]["frame_index"] == 0
    assert out[1]["frame_index"] == 1
    assert out[2]["frame_index"] == 1  # whole segment, one frame, by midpoint


def test_five_frame_annotation_hard_gate():
    """§7/§16.7: annotate 5 frames with the segment range each should own; assert each
    segment's midpoint lands in its frame's interval. Pure pass/fail, no judgment."""
    fr = _frames([2.0, 6.0, 10.0, 14.0, 18.0])
    anchors = alignment.build_anchors(scene_cuts=[4.0, 8.0, 12.0, 16.0], speaker_turns=[],
                                      silence_gaps=[], caption_cues=[], duration=20.0)
    # expected: segment k belongs under frame k
    segs = []
    for k, mid in enumerate([2.0, 6.0, 10.0, 14.0, 18.0]):
        segs.append({"index": k, "start_s": mid - 0.5, "end_s": mid + 0.5,
                     "speaker": None, "text": f"s{k}", "frame_index": None})
    out = alignment.align(fr, segs, anchors, duration=20.0)
    assert [s["frame_index"] for s in out] == [0, 1, 2, 3, 4]


def test_determinism_same_input_same_output():
    fr = _frames([1.0, 5.0, 9.0])
    anchors = alignment.build_anchors([3.0], [], [], [], duration=10.0)
    segs = [{"index": 0, "start_s": 4.0, "end_s": 6.0, "speaker": None, "text": "x", "frame_index": None}]
    a = alignment.align(fr, list(segs), anchors, duration=10.0)
    b = alignment.align(fr, list(segs), anchors, duration=10.0)
    assert [s["frame_index"] for s in a] == [s["frame_index"] for s in b]


def test_no_anchor_stretch_splits_evenly():
    # only bookend anchors → all frames in one cell → even split by frame midpoints
    fr = _frames([2.0, 8.0])
    anchors = alignment.build_anchors([], [], [], [], duration=10.0)
    intervals = alignment.frame_intervals(fr, anchors, duration=10.0)
    assert intervals[0] == [0, 0.0, 5.0]
    assert intervals[1] == [1, 5.0, 10.0]


def test_single_frame_owns_everything():
    # golden (Codex F4): one frame owns the whole timeline; every segment maps to frame 0
    fr = _frames([4.0])
    anchors = alignment.build_anchors(scene_cuts=[2.0, 7.0], speaker_turns=[], silence_gaps=[],
                                      caption_cues=[], duration=12.0)
    assert alignment.frame_intervals(fr, anchors, duration=12.0) == [[0, 0.0, 12.0]]
    segs = [
        {"index": 0, "start_s": 0.0, "end_s": 1.0, "speaker": None, "text": "a", "frame_index": None},
        {"index": 1, "start_s": 5.0, "end_s": 6.0, "speaker": None, "text": "b", "frame_index": None},
        {"index": 2, "start_s": 11.0, "end_s": 12.0, "speaker": None, "text": "c", "frame_index": None},
    ]
    out = alignment.align(fr, segs, anchors, duration=12.0)
    assert [s["frame_index"] for s in out] == [0, 0, 0]


def test_caption_only_anchors_golden():
    # golden (Codex F4): captions path — anchors built ONLY from caption_cues (no scene cuts, no
    # speaker turns, no silence). Two frames; the expected frame_index mapping is written by HAND
    # (not derived from frame_intervals), so a regression in interval construction is caught.
    fr = _frames([2.0, 8.0])
    anchors = alignment.build_anchors(scene_cuts=[], speaker_turns=[], silence_gaps=[],
                                      caption_cues=[0.0, 5.0, 10.0], duration=10.0)
    segs = [
        {"index": 0, "start_s": 0.0, "end_s": 2.0, "speaker": None, "text": "s0", "frame_index": None},  # mid 1.0
        {"index": 1, "start_s": 3.0, "end_s": 5.0, "speaker": None, "text": "s1", "frame_index": None},  # mid 4.0
        {"index": 2, "start_s": 5.0, "end_s": 7.0, "speaker": None, "text": "s2", "frame_index": None},  # mid 6.0
        {"index": 3, "start_s": 8.0, "end_s": 10.0, "speaker": None, "text": "s3", "frame_index": None},  # mid 9.0
    ]
    out = alignment.align(fr, segs, anchors, duration=10.0)
    # frame 0 owns [0,5), frame 1 owns [5,10]; by hand: mids 1,4 → frame 0; mids 6,9 → frame 1
    assert [s["frame_index"] for s in out] == [0, 0, 1, 1]
