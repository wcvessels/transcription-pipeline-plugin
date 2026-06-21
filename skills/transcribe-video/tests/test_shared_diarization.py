import importlib
import sys
from pathlib import Path

import pytest

SHARED_SCRIPTS = (Path.home() / ".claude" / "skills" / "_shared" / "diarization" / "scripts")


def test_shared_diarization_dir_exists():
    assert SHARED_SCRIPTS.is_dir(), f"promoted diarization clone not found at {SHARED_SCRIPTS}"
    assert (SHARED_SCRIPTS / "diarize_pipeline.py").is_file()


def test_clone_exposes_expected_interface():
    if str(SHARED_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SHARED_SCRIPTS))
    dp = importlib.import_module("diarize_pipeline")
    for name in ("get_pipeline", "annotation_to_dataframe", "unwrap_annotation", "ensure_models"):
        assert hasattr(dp, name), f"diarize_pipeline missing {name}"


def test_model_weights_present_after_move():
    models = SHARED_SCRIPTS.parent / "models" / "diarization"
    assert (models / "segmentation-3.0.bin").is_file()
    assert (models / "wespeaker-resnet34-lm.bin").is_file()
    assert (models / "config.yaml").is_file()
