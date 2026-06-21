#!/usr/bin/env bash
# Refresh the plugin's skill snapshot from the canonical dev source (~/.claude/skills).
#
# Snapshot model: dev is canonical; the plugin is a versioned copy. This re-copies the three
# subtrees, prunes weights/caches/corpus, and applies the dev->plugin transforms on SKILL.md
# (absolute dev paths -> ${CLAUDE_PLUGIN_ROOT}; "GTX 1080" host claims -> hardware-neutral).
# ponytail: a copy + a few seds, not a submodule. Bump plugin.json version after a sync.
set -euo pipefail
SRC="${1:-$HOME/.claude/skills}"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for s in transcribe-audio transcribe-video _shared; do
  rm -rf "$DST/skills/$s"
  cp -r "$SRC/$s" "$DST/skills/"
done

# prune: caches, diarization weights (refetch + sha256-verify on first use), test corpus, empties
find "$DST/skills" -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$DST/skills/_shared/diarization/models" "$DST/skills/transcribe-video/corpus"
rmdir "$DST/skills/transcribe-audio/models" 2>/dev/null || true
find "$DST/skills" -name '*.bin' -delete 2>/dev/null || true

# dev -> plugin transforms (single-quoted so ${CLAUDE_PLUGIN_ROOT} is written LITERALLY)
for skill in transcribe-audio transcribe-video; do
  f="$DST/skills/$skill/SKILL.md"
  sed -i 's|C:/Users/Will/.claude/skills/|${CLAUDE_PLUGIN_ROOT}/skills/|g' "$f"
  sed -i 's|GPU acceleration on GTX 1080|GPU auto-detected (CPU fallback)|' "$f"
  sed -i 's|GPU acceleration via NVIDIA GeForce GTX 1080|GPU auto-detected (CPU fallback)|' "$f"
  sed -i 's|GTX 1080 (GPU): ~10-15x realtime.*|GPU: ~5-15x realtime depending on card and diarization|' "$f"
done

echo "Synced from $SRC."
echo "Next: re-run tests in skills/*/tests, run scripts/check-environment.py, bump plugin.json version."
echo "If dev SKILL.md GPU/perf wording changed, re-check the neutralizing seds above still match."
