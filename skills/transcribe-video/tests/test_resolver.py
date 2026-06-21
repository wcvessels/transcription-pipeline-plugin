import subprocess
from pathlib import Path

import pytest
import resolver


RESERVED_CONNECTOR_HINTS = ["m365", "box", "gong", "fireflies"]


def _make_tiny_mp4(path: Path):
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", "2", str(path),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_classify_local_file(tmp_path):
    f = tmp_path / "clip.mp4"
    _make_tiny_mp4(f)
    assert resolver.classify_source(str(f), source_hint=None) == "file"


def test_classify_public_url():
    assert resolver.classify_source("https://www.youtube.com/watch?v=abc", source_hint=None) == "yt-dlp"


def test_unrecognized_source_fails_with_supported_list():
    with pytest.raises(resolver.UnsupportedSourceError) as exc:
        resolver.classify_source("not-a-path-or-url", source_hint=None)
    assert "local file" in str(exc.value).lower()
    assert "youtube" in str(exc.value).lower() or "url" in str(exc.value).lower()


@pytest.mark.parametrize("hint", RESERVED_CONNECTOR_HINTS)
def test_reserved_connector_hint_errors_cleanly(hint):
    with pytest.raises(resolver.ReservedFeatureError) as exc:
        resolver.classify_source("https://example.com/x", source_hint=hint)
    assert "A1.x" in str(exc.value)
    assert "not available yet" in str(exc.value)


def test_functional_hints_force_branch():
    assert resolver.classify_source("anything", source_hint="file") == "file"
    assert resolver.classify_source("anything", source_hint="url") == "yt-dlp"


def test_probe_metadata_on_local_file(tmp_path):
    f = tmp_path / "clip.mp4"
    _make_tiny_mp4(f)
    md = resolver.probe_metadata(f)
    assert md["width"] == 320 and md["height"] == 240
    assert md["duration_s"] == pytest.approx(2.0, abs=0.3)
    assert md["codec"]  # non-empty
    assert md["fps"] > 0


def test_resolve_local_returns_path_and_metadata(tmp_path):
    f = tmp_path / "clip.mp4"
    _make_tiny_mp4(f)
    workdir = tmp_path / "work"
    workdir.mkdir()
    local_path, md = resolver.resolve(str(f), workdir, source_hint=None)
    assert local_path == f.resolve()
    assert md["type"] == "file" and md["uri"] == str(f.resolve())
    assert md["download_s"] == 0.0  # local file: no download


def test_non_video_file_rejected(tmp_path):
    # R3 P3: an existing but non-video file must be rejected cleanly at classification, not treated
    # as 'file' and then crash inside ffprobe.
    f = tmp_path / "notes.txt"
    f.write_text("not a video", encoding="utf-8")
    with pytest.raises(resolver.UnsupportedSourceError):
        resolver.classify_source(str(f), source_hint=None)
