#!/usr/bin/env bash
# Fetch EuRoC MAV sequences (ASL dataset format) into data/raw/.
# Usage: ./data/fetch_euroc.sh [SEQ ...]   (default: V1_01)
# Source: ETHZ ASL, https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

SEQS=("$@")
if [ ${#SEQS[@]} -eq 0 ]; then SEQS=(V1_01); fi

base_for() {
  case "$1" in
    MH_0*) echo "machine_hall" ;;
    V1_0*) echo "vicon_room1" ;;
    V2_0*) echo "vicon_room2" ;;
    *) echo "unknown sequence $1 (want MH_01..05, V1_01..03, V2_01..03)" >&2; exit 1 ;;
  esac
}

for seq in "${SEQS[@]}"; do
  group=$(base_for "$seq")
  out="data/raw/${seq}.zip"
  if [ -f "$out" ] && unzip -tqq "$out" >/dev/null 2>&1; then
    echo "$seq: already present and intact, skipping"
    continue
  fi
  url="https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/${group}/${seq}/${seq}.zip"
  echo "$seq: $url"
  curl -fL --retry 3 --retry-delay 5 -o "$out" "$url"
  unzip -tqq "$out" >/dev/null
  echo "$seq: OK ($(du -h "$out" | cut -f1))"
done
