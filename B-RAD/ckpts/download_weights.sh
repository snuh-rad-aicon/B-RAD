#!/usr/bin/env bash
# Download the B-RAD inference checkpoints into this folder (B-RAD/ckpts/).
#
# The weights on Hugging Face are gated. Request access at
# https://huggingface.co/hanjang/B-RAD, then run `hf auth login`
# or `export HF_TOKEN=hf_...` before running this script.
#
#   bash B-RAD/ckpts/download_weights.sh
#   JOBS=1 bash B-RAD/ckpts/download_weights.sh    # one connection per file
set -euo pipefail

CKPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CKPT_DIR"

HF_REPO="${HF_REPO:-hanjang/B-RAD}"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"
ULTRASAM_URL="https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"

# Connections per file. The UltraSam host throttles each connection,
# so splitting a file into parallel ranges turns hours into minutes.
JOBS="${JOBS:-16}"
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ] || [ "$JOBS" -gt 32 ]; then
  echo "[warn] JOBS must be between 1 and 32, using 16"
  JOBS=16
fi

# Use HF_TOKEN if set, otherwise the token saved by `hf auth login`.
TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$TOKEN_FILE")"
fi
HF_TOKEN="${HF_TOKEN:-}"


# Attach the token for huggingface.co only, never for the UltraSam host.
curl_with_auth() {
  local url="$1"
  shift
  if [[ "$url" == https://huggingface.co/* ]] && [ -n "$HF_TOKEN" ]; then
    curl --silent --show-error --fail --location --http1.1 \
         --header "Authorization: Bearer ${HF_TOKEN}" "$@" "$url"
  else
    curl --silent --show-error --fail --location --http1.1 "$@" "$url"
  fi
}

file_size() {
  if [ -f "$1" ]; then
    stat -c %s "$1" 2>/dev/null || stat -f %z "$1"
  else
    echo 0
  fi
}

remote_size() {
  curl_with_auth "$1" --head \
    | awk 'tolower($1) == "content-length:" { size = $2 } END { gsub(/\r/, "", size); print size + 0 }'
}

download_range() {
  local url="$1" part="$2" start="$3" end="$4"
  local need=$(( end - start + 1 ))
  local have attempt

  for (( attempt = 1; attempt <= 60; attempt++ )); do
    have="$(file_size "$part")"
    if [ "$have" -ge "$need" ]; then
      return 0
    fi
    curl_with_auth "$url" --speed-time 30 --speed-limit 1000 \
      --range "$(( start + have ))-${end}" >> "$part" 2>/dev/null || true
  done
}

# Split the file into JOBS ranges under .parts/<name>/, download them in
# parallel, then join them. Re-running the script resumes unfinished parts.
download_parallel() {
  local url="$1" name="$2" total="$3"
  local part_dir=".parts/${name}"
  local chunk=$(( (total + JOBS - 1) / JOBS ))
  local i start end

  mkdir -p "$part_dir"
  for (( i = 0; i < JOBS; i++ )); do
    start=$(( i * chunk ))
    end=$(( start + chunk - 1 ))
    if [ "$end" -ge "$total" ]; then
      end=$(( total - 1 ))
    fi
    download_range "$url" "${part_dir}/part_$(printf '%03d' "$i")" "$start" "$end" &
  done
  wait

  cat "${part_dir}"/part_* > "$name"
  rm -rf "$part_dir"
}

get_checkpoint() {
  local name="$1" url="$2"
  local total actual

  if ! total="$(remote_size "$url")" || [ "$total" -le 0 ]; then
    echo "[error] could not read the size of ${name} from ${url}" >&2
    if [[ "$url" == https://huggingface.co/* ]]; then
      echo "[error] check that your account has access to https://huggingface.co/${HF_REPO} and that the token is valid" >&2
    fi
    exit 1
  fi

  if [ "$(file_size "$name")" = "$total" ]; then
    echo "[skip] ${name} already complete (${total} bytes)"
    return 0
  fi

  echo "[get] ${name} (${total} bytes, ${JOBS} connections)"
  download_parallel "$url" "$name" "$total"

  actual="$(file_size "$name")"
  if [ "$actual" != "$total" ]; then
    echo "[error] ${name} is ${actual} of ${total} bytes, run the script again to resume" >&2
    exit 1
  fi
  echo "[ok] ${name}"
}


echo "[info] repo: ${HF_REPO}"
echo "[info] target: ${CKPT_DIR}"
echo "[info] connections per file: ${JOBS}"
if [ -n "$HF_TOKEN" ]; then
  echo "[info] Hugging Face token found"
else
  echo "[warn] no Hugging Face token found, request access at https://huggingface.co/${HF_REPO} and run 'hf auth login' or export HF_TOKEN"
fi

# A single request to config.json is what the Hub counts as one download.
if curl_with_auth "${HF_BASE}/config.json" --output /dev/null; then
  echo "[ok] config.json"
else
  echo "[warn] could not fetch config.json, continuing"
fi

get_checkpoint "b_rad_clip.pt" "${HF_BASE}/b_rad_clip.pt"
get_checkpoint "b_rad_full.pt" "${HF_BASE}/b_rad_full.pt"
get_checkpoint "b_rad_det.pt"  "${HF_BASE}/b_rad_det.pt"
get_checkpoint "UltraSam.pth"  "${ULTRASAM_URL}"    # external, CC BY-NC-SA 4.0 (non-commercial)

echo "[done] checkpoints in ${CKPT_DIR}"
ls -lh b_rad_clip.pt b_rad_full.pt b_rad_det.pt UltraSam.pth
