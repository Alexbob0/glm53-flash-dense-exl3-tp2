#!/bin/bash
# Builds a synthetic HF-cache entry over the dense-overlay pack, using
# RELATIVE symlinks so the entry resolves both on the host and inside the
# serving container (which mounts the HF cache at /root/.cache/huggingface).
#
# Usage:
#   HF_CACHE=~/hf SRC_SNAPSHOT=<path-to-TR3-4bpw-snapshot> \
#   OVERLAY=<dense_overlay.py --out dir> ./make-cache-entry.sh
set -euo pipefail
HF_CACHE="${HF_CACHE:?point to your HF cache root (contains hub/)}"
SRC_SNAPSHOT="${SRC_SNAPSHOT:?path to the source pack snapshot dir}"
OVERLAY="${OVERLAY:?path to the dense-overlay output dir}"
ENTRY_NAME="${ENTRY_NAME:-models--local--glm53-dense-K6}"
SNAP_NAME="${SNAP_NAME:-k6dense}"

SRC_REL="../../../$(basename "$(dirname "$(dirname "$SRC_SNAPSHOT")")")/snapshots/$(basename "$SRC_SNAPSHOT")"
E="$HF_CACHE/hub/$ENTRY_NAME"
[ -f "$OVERLAY"/dense-exl3-*.safetensors ] || { echo "overlay pack missing"; exit 1; }
rm -rf "$E"
mkdir -p "$E/snapshots/$SNAP_NAME" "$E/refs"
echo -n "$SNAP_NAME" > "$E/refs/main"
cd "$E/snapshots/$SNAP_NAME"
for f in "$SRC_SNAPSHOT"/*; do
  ln -s "$SRC_REL/$(basename "$f")" "$(basename "$f")"
done
rm -f model.safetensors.index.json config.json
ln "$OVERLAY"/dense-exl3-*.safetensors .
cp "$OVERLAY/model.safetensors.index.json" .
cp "$OVERLAY/config.json" .
echo "entry: $E (shards: $(find "$E/snapshots" -name '*.safetensors' | wc -l))"
