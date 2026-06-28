from PIL import Image

import curated_output as co


def _segments_whisperx():
    return [
        {"index": 0, "start_s": 0.0, "end_s": 3.0, "speaker": "SPEAKER_00", "text": "hello", "frame_index": 0},
        {"index": 1, "start_s": 3.0, "end_s": 6.0, "speaker": "SPEAKER_01", "text": "world", "frame_index": 1},
    ]


def _segments_captions():
    return [
        {"index": 0, "start_s": 0.0, "end_s": 3.0, "speaker": None, "text": "hello there", "frame_index": 0},
        {"index": 1, "start_s": 3.0, "end_s": 6.0, "speaker": None, "text": "and welcome", "frame_index": 1},
    ]


def _frame_records(tmp_path):
    recs = []
    for i, (ts, sharp) in enumerate([(0.0, 142.5), (5.0, 99.0)]):
        f = tmp_path / f"frame_{i+1:04d}.jpg"
        Image.new("RGB", (160, 90), (i * 40, 60, 200)).save(f)
        recs.append({"index": i, "timestamp_s": ts, "file": f.name, "is_scene_cut": i == 0,
                     "phash": "0", "sharpness": sharp})
    return recs


# ---- transcript (BOTH paths) ----

def test_transcript_whisperx_has_speaker_labels(tmp_path):
    out = tmp_path / "demo_transcript.txt"
    co.write_transcript(_segments_whisperx(), out)
    text = out.read_text(encoding="utf-8")
    assert "SPEAKER_00" in text and "SPEAKER_01" in text
    assert "hello" in text and "world" in text


def test_transcript_captions_has_no_speaker_labels(tmp_path):
    out = tmp_path / "demo_transcript.txt"
    co.write_transcript(_segments_captions(), out)
    text = out.read_text(encoding="utf-8")
    assert "SPEAKER_" not in text and "[" not in text
    assert "hello there" in text and "and welcome" in text


# ---- frames index ----

def test_frames_index_lists_each_frame_with_sharpness(tmp_path):
    recs = _frame_records(tmp_path)
    out = tmp_path / "demo_frames.md"
    co.write_frames_index(recs, "demo_frames", out)
    text = out.read_text(encoding="utf-8")
    assert "demo_frames/frame_0001.jpg" in text and "demo_frames/frame_0002.jpg" in text
    assert "142.5" in text                       # sharpness shown
    assert text.count("frame_") >= 2
