from pathlib import Path

import captions

FIXTURE = Path(__file__).parent / "fixtures" / "sample.vtt"


def test_parse_vtt_returns_cues_with_times():
    cues = captions.parse_vtt(FIXTURE.read_text(encoding="utf-8"))
    assert len(cues) == 4
    assert cues[0]["start_s"] == 0.0 and cues[0]["end_s"] == 2.0
    assert cues[3]["start_s"] == 6.0


def test_captions_captured_verbatim():
    cues = captions.parse_vtt(FIXTURE.read_text(encoding="utf-8"))
    segs = captions.to_segments(cues)
    # VERBATIM: every cue is preserved one-for-one, no rolling-overlap collapse (de-overlap is A2)
    assert len(segs) == len(cues)
    joined = " ".join(s["text"] for s in segs)
    # the rolling-window overlap text survives verbatim — "to the demo" appears in BOTH cue 1 and cue 2
    assert joined.count("to the demo") == 2
    assert joined.count("today we will") == 2
    assert "settings panel" in joined


def test_caption_segments_have_no_speaker():
    cues = captions.parse_vtt(FIXTURE.read_text(encoding="utf-8"))
    segs = captions.to_segments(cues)
    assert all(s["speaker"] is None for s in segs)
    assert all("start_s" in s and "end_s" in s and "text" in s for s in segs)


def test_to_segments_indexes_and_shapes(tmp_path):
    cues = captions.parse_vtt(FIXTURE.read_text(encoding="utf-8"))
    segs = captions.to_segments(cues)
    assert segs[0]["index"] == 0
    assert all(s["frame_index"] is None for s in segs)  # alignment fills this later


def test_caption_cue_boundaries_extracted():
    cues = captions.parse_vtt(FIXTURE.read_text(encoding="utf-8"))
    bounds = captions.cue_boundaries(cues)
    assert 0.0 in bounds and 8.0 in bounds
    assert bounds == sorted(bounds)
