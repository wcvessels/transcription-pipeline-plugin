#!/usr/bin/env python
"""Environment self-test for the transcription plugin.

Exit 0 if the machine is ready to transcribe, non-zero if a CRITICAL dependency is missing.
GPU is a warning, not a failure (CPU works, just slower). Used by /transcribe-setup and for
debugging. Run: python scripts/check-environment.py
"""
import importlib
import shutil
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
# reuse the skill's real GPU-class logic so the report matches what transcription will pick
sys.path.insert(0, str(PLUGIN_ROOT / "skills" / "transcribe-video" / "scripts"))

_ok = True


def check(label, passed, detail="", critical=True):
    global _ok
    mark = "OK  " if passed else ("FAIL" if critical else "WARN")
    if not passed and critical:
        _ok = False
    print(f"[{mark}] {label}" + (f" - {detail}" if detail else ""))


check(f"Python {sys.version_info.major}.{sys.version_info.minor} (>= 3.10)",
      sys.version_info >= (3, 10), sys.version.split()[0])

ff = shutil.which("ffmpeg")
check("ffmpeg on PATH", ff is not None, ff or "not found - install ffmpeg, then restart the shell")

for mod in ["torch", "whisperx", "faster_whisper", "yt_dlp", "imagehash", "PIL", "jsonschema", "numpy", "pandas"]:
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except Exception as e:  # ImportError or a broken install
        check(f"import {mod}", False, str(e).splitlines()[0])

# GPU is optional: CPU works but is much slower
try:
    import torch
    from transcription import auto_compute_type, auto_batch_size, detect_gpu_name, detect_vram_gb
    if torch.cuda.is_available():
        name, vram = detect_gpu_name(), detect_vram_gb()
        check(f"CUDA GPU: {name}", True,
              f"compute_type={auto_compute_type(name)}, batch_size={auto_batch_size(vram)}, {vram:.0f} GB VRAM",
              critical=False)
    else:
        check("CUDA GPU", False, "none detected - will run on CPU (much slower)", critical=False)
except Exception as e:
    check("GPU probe", False, str(e).splitlines()[0], critical=False)

print()
print("READY" if _ok else "NOT READY - fix the FAIL items above")
sys.exit(0 if _ok else 1)
