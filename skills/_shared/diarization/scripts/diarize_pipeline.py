#!/usr/bin/env python
"""Local pyannote speaker diarization: verified weights, no HF token.

Auto-fetches the pyannote 3.1 pipeline weights on first use and verifies
every byte against the official repos' published sha256 (served anonymously
by the HF API even for the gated file). After first fetch, fully offline.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = SKILL_DIR / "models" / "diarization"
CONFIG_FILE = MODELS_DIR / "config.yaml"

SEGMENTATION_FILE = "segmentation-3.0.bin"
EMBEDDING_FILE = "wespeaker-resnet34-lm.bin"

# Official sha256 values, read anonymously from the HF API (?blobs=true).
# Segmentation's repo is gated but its hash is not; mirrors below were
# verified bit-identical to official on 2026-06-10.
SEGMENTATION_SHA256 = "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298"
EMBEDDING_SHA256 = "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"

# License-compliant mirrors first: tensorlake + ubitec both ship the upstream MIT LICENSE
# (Copyright (c) 2023 CNRS); ivrit-ai carries no license/NOTICE, so it is last-resort fallback
# only. All three are sha256-pinned to the bit-identical official file (verified 2026-06-10).
SEGMENTATION_SOURCES = [
    "https://huggingface.co/tensorlake/segmentation-3.0/resolve/main/pytorch_model.bin",
    "https://huggingface.co/ubitec/pyannote-segmentation-3.0/resolve/main/pytorch_model.bin",
    "https://huggingface.co/ivrit-ai/pyannote-segmentation-3.0/resolve/main/pytorch_model.bin",
]
EMBEDDING_SOURCES = [
    "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/pytorch_model.bin",
]


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download(url, dest):
    print(f"[diarize] downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "diarize-local/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)


def ensure_file(dest, sources, expected_sha256, download=_download):
    """Guarantee dest exists with the expected sha256. Returns True if fetched.

    Tries sources in order. An unverified file never occupies dest -- downloads
    go to a .part file, which is hashed before being moved into place with
    Path.replace. The downloader must write the given path or raise.

    Not concurrency-safe on first fetch (fixed .part name); concurrent callers
    self-heal on retry.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256_of(dest) == expected_sha256:
            return False
        print(f"[diarize] {dest.name} fails verification, re-fetching", file=sys.stderr)
        dest.unlink()

    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in sources:
        part.unlink(missing_ok=True)
        try:
            download(url, part)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            part.unlink(missing_ok=True)
            continue
        actual = sha256_of(part)
        if actual == expected_sha256:
            part.replace(dest)
            print(f"[diarize] verified {dest.name} sha256={actual[:12]}...", file=sys.stderr)
            return True
        errors.append(f"{url}: sha256 mismatch (got {actual})")
        part.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not obtain a verified copy of {dest.name} "
        f"(expected sha256 {expected_sha256}).\nSources tried:\n  "
        + "\n  ".join(errors)
        + "\nIf offline, reconnect and retry; auto-fetch re-verifies on the next run."
    )


CANONICAL_CONFIG = """\
# Canonical pyannote/speaker-diarization-3.1 hyperparameters.
# Auto-generated from CANONICAL_CONFIG in diarize_pipeline.py — edits to this file are reverted on next run; edit the constant instead.
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    segmentation: segmentation-3.0.bin
    embedding: wespeaker-resnet34-lm.bin
    clustering: AgglomerativeClustering
    embedding_exclude_overlap: true
    segmentation_batch_size: 32
    embedding_batch_size: 32
params:
  segmentation:
    min_duration_off: 0.0
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
"""


def ensure_models(download=_download):
    """Fetch-and-verify both weight files and keep config.yaml in sync with
    CANONICAL_CONFIG (rewrites on drift). Idempotent; no network when files
    already verify."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_file(MODELS_DIR / SEGMENTATION_FILE, SEGMENTATION_SOURCES,
                SEGMENTATION_SHA256, download=download)
    ensure_file(MODELS_DIR / EMBEDDING_FILE, EMBEDDING_SOURCES,
                EMBEDDING_SHA256, download=download)
    if not CONFIG_FILE.exists() or CONFIG_FILE.read_text(encoding="utf-8") != CANONICAL_CONFIG:
        CONFIG_FILE.write_text(CANONICAL_CONFIG, encoding="utf-8")
    return MODELS_DIR


def unwrap_annotation(result):
    """pyannote 4.x pipelines return DiarizeOutput (annotation at
    .speaker_diarization); 3.x returned Annotation directly. Accept both.

    We use the standard (possibly overlapping) annotation, not
    exclusive_speaker_diarization, for parity with the 3.1 token pipeline.
    """
    from pyannote.core import Annotation

    if isinstance(result, Annotation):
        return result
    inner = getattr(result, "speaker_diarization", None)
    if isinstance(inner, Annotation):
        return inner
    raise TypeError(f"cannot extract Annotation from {type(result).__name__}")


def annotation_to_dataframe(annotation):
    """pyannote Annotation or pipeline DiarizeOutput -> DataFrame with
    start/end/speaker columns, the shape whisperx.assign_word_speakers expects.
    Accepts either a bare Annotation or a DiarizeOutput (pyannote 4.x)."""
    annotation = unwrap_annotation(annotation)
    import pandas as pd

    rows = [
        {"segment": segment, "label": label, "speaker": speaker,
         "start": segment.start, "end": segment.end}
        for segment, label, speaker in annotation.itertracks(yield_label=True)
    ]
    df = pd.DataFrame(rows, columns=["segment", "label", "speaker", "start", "end"])
    return df.sort_values("start").reset_index(drop=True)


def get_pipeline(device=None):
    """Ready-to-run local diarization pipeline. Fetches+verifies models on
    first use; afterwards loads fully offline. No token, ever.

    Call as pipeline({"waveform": (1, T) float32 tensor, "sample_rate": sr},
    num_speakers=/min_speakers=/max_speakers=...); pass the result through
    unwrap_annotation().
    """
    import torch
    import yaml
    from pyannote.audio import Model
    from pyannote.audio.core.plda import PLDA
    from pyannote.audio.pipelines import SpeakerDiarization

    ensure_models()
    cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    p = cfg["pipeline"]["params"]

    segmentation = Model.from_pretrained(str(MODELS_DIR / p["segmentation"]))
    # Embedding must be a Model OBJECT: a "wespeaker" string path routes to
    # pyannote's ONNX loader, which cannot read a .bin torch checkpoint.
    embedding = Model.from_pretrained(str(MODELS_DIR / p["embedding"]))

    pipeline = SpeakerDiarization(
        segmentation=segmentation,
        embedding=embedding,
        clustering=p["clustering"],
        embedding_exclude_overlap=p["embedding_exclude_overlap"],
        segmentation_batch_size=p["segmentation_batch_size"],
        embedding_batch_size=p["embedding_batch_size"],
        # 4.0.4 defaults plda to the gated community-1 repo and fetches it at
        # construction. AgglomerativeClustering never reads PLDA (only VBx
        # does); an empty stub skips the gated fetch entirely.
        plda=PLDA.__new__(PLDA),
    )
    pipeline.instantiate(cfg["params"])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    return pipeline


if __name__ == "__main__":
    ensure_models()
    print(f"[diarize] models ready in {MODELS_DIR}", file=sys.stderr)
