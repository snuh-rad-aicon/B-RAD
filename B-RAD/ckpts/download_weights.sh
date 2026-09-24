#!/usr/bin/env bash
# Download the B-RAD inference checkpoints into this folder (B-RAD/ckpts/).
# Fetched with JOBS parallel range requests; the UltraSam host throttles per
# connection, so the default 16 turns hours into minutes. JOBS=1 for one stream.
set -euo pipefail

CKPTS="$(cd "$(dirname "$0")" && pwd)"
cd "$CKPTS"

HF_REPO="${HF_REPO:-hanjang/B-RAD}"
JOBS="${JOBS:-16}"
case "$JOBS" in ''|*[!0-9]*) JOBS=16 ;; esac
[ "$JOBS" -ge 1 ] && [ "$JOBS" -le 32 ] || JOBS=16

HF="https://huggingface.co/${HF_REPO}/resolve/main"
# UltraSam is external and licensed CC BY-NC-SA 4.0 (non-commercial only).
# https://github.com/CAMMA-public/UltraSam
ULTRASAM="https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"

FILES=(
  "b_rad_clip.pt:${HF}/b_rad_clip.pt"
  "b_rad_full.pt:${HF}/b_rad_full.pt"
  "b_rad_det.pt:${HF}/b_rad_det.pt"
  "UltraSam.pth:${ULTRASAM}"
)

size_of() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }

remote_size() {
  curl --silent --show-error --fail --location --http1.1 --head "$1" \
    | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v+0}'
}

# JOBS parallel ranges into .parts/<name>/, then concatenate. Each part appends
# from wherever it stopped, so re-running resumes instead of restarting.
fetch() {
  local url="$1" name="$2" total="$3" dir=".parts/$2" i
  local chunk=$(( (total + JOBS - 1) / JOBS ))
  mkdir -p "$dir"

  for i in $(seq 0 $(( JOBS - 1 ))); do
    (
      start=$(( i * chunk )); end=$(( start + chunk - 1 ))
      [ "$end" -ge "$total" ] && end=$(( total - 1 ))
      part="${dir}/part_$(printf '%03d' "$i")"
      for _ in $(seq 1 60); do
        have=$(size_of "$part")
        [ "$have" -ge $(( end - start + 1 )) ] && break
        curl --silent --show-error --fail --location --http1.1 \
             --speed-time 30 --speed-limit 1000 \
             --range "$(( start + have ))-${end}" "$url" >> "$part" 2>/dev/null || true
      done
    ) &
  done
  wait

  cat "${dir}"/part_* > "$name"
  rm -rf "$dir"
}

for entry in "${FILES[@]}"; do
  name="${entry%%:*}"; url="${entry#*:}"
  total="$(remote_size "$url")"
  if [ "$total" -le 0 ]; then
    echo "ERROR: could not read the size of $name from $url" >&2
    exit 1
  fi

  if [ "$(size_of "$name")" = "$total" ]; then
    echo "[skip] $name already complete ($total bytes)"
    continue
  fi

  echo "[get] $name ($total bytes, ${JOBS} connections)"
  fetch "$url" "$name" "$total"

  actual="$(size_of "$name")"
  if [ "$actual" != "$total" ]; then
    echo "ERROR: $name is $actual of $total bytes. Re-run to resume." >&2
    exit 1
  fi
  echo "[ok] $name complete"
done

echo "Done. Checkpoints in $CKPTS/"
