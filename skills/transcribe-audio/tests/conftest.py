import sys
from pathlib import Path

# Put scripts/ on sys.path so tests can `import transcribe`. transcribe.py imports torch/whisperx
# lazily (inside transcribe()), so importing the module here is GPU-free.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
