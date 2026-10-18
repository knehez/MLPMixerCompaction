#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [smoke|full] [download|no-download]"
  echo 'Environment: MIXER_COLAB_GPU=G4, MIXER_COLAB_TIMEOUT=86400'
  echo 'Optional: MIXER_CIFAR10_URL=<mirror archive URL>'
  echo 'Sweep: MIXER_LEVELS=0.60,0.65,0.70,0.75,0.80,0.85 MIXER_SEEDS=123,456,789'
  echo 'Training: MIXER_BASELINE_EPOCHS=80 MIXER_FINETUNE_EPOCHS=30 MIXER_PATIENCE=6'
  echo 'Depth: MIXER_DEPTH_REMOVALS=1,2 MIXER_MIN_BLOCKS=1 MIXER_COMBINED_WIDTH=0.75'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
MODE="${1:-smoke}"
TRANSFER="${2:-download}"
if (( $# > 2 )) || [[ "$MODE" != smoke && "$MODE" != full ]] ||
   [[ "$TRANSFER" != download && "$TRANSFER" != no-download ]]; then
  usage >&2; exit 2
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="lottery-mixer-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
GPU="${MIXER_COLAB_GPU:-G4}"
TIMEOUT="${MIXER_COLAB_TIMEOUT:-86400}"
DEPTH_REMOVALS="${MIXER_DEPTH_REMOVALS-1,2}" # An explicit empty value disables depth experiments.
SOURCE="$ROOT/mlp_mixer_cifar10.py"
RUNNER="$ROOT/scripts/colab_run_mixer.py"
RESULT="$ROOT/artifacts/${SESSION}_results.zip"
if [[ ! "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then echo 'Invalid MIXER_COLAB_TIMEOUT' >&2; exit 2; fi
command -v colab >/dev/null 2>&1 || {
  echo 'Missing colab CLI. Install in WSL: uv tool install google-colab-cli' >&2
  exit 1
}
[[ -f "$SOURCE" && -f "$RUNNER" ]] || { echo 'Missing source or remote runner.' >&2; exit 1; }
if [[ "$TRANSFER" == download ]]; then mkdir -p "$ROOT/artifacts"; fi

cleanup() {
  if ! colab stop -s "$SESSION"; then
    echo "Could not confirm session shutdown. Run: colab stop -s $SESSION" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting session=$SESSION GPU=$GPU mode=$MODE"
colab new -s "$SESSION" --gpu "$GPU"
colab upload -s "$SESSION" "$SOURCE" /content/mlp_mixer_cifar10.py
ARCHIVE=1
[[ "$TRANSFER" == download ]] || ARCHIVE=0
exec_status=0
colab exec --timeout "$TIMEOUT" -s "$SESSION" \
  --env "MIXER_MODE=$MODE" --env "MIXER_ARCHIVE_RESULTS=$ARCHIVE" \
  --env "MIXER_CIFAR10_URL=${MIXER_CIFAR10_URL:-}" \
  --env "MIXER_LEVELS=${MIXER_LEVELS:-0.60,0.65,0.70,0.75,0.80,0.85}" \
  --env "MIXER_SEEDS=${MIXER_SEEDS:-123,456,789}" \
  --env "MIXER_DATA_ROOT=/content/mixer_data" \
  --env "MIXER_BASELINE_EPOCHS=${MIXER_BASELINE_EPOCHS:-80}" \
  --env "MIXER_FINETUNE_EPOCHS=${MIXER_FINETUNE_EPOCHS:-30}" \
  --env "MIXER_PATIENCE=${MIXER_PATIENCE:-6}" \
  --env "MIXER_DEPTH_REMOVALS=$DEPTH_REMOVALS" \
  --env "MIXER_MIN_BLOCKS=${MIXER_MIN_BLOCKS:-1}" \
  --env "MIXER_COMBINED_WIDTH=${MIXER_COMBINED_WIDTH:-0.75}" \
  -f "$RUNNER" || exec_status=$?

download_status=0
if [[ "$TRANSFER" == download ]]; then
  colab download -s "$SESSION" /content/mixer_results.zip "$RESULT" || download_status=$?
fi
if (( exec_status != 0 )); then
  echo "Mixer run failed (exit $exec_status)." >&2
  if [[ "$TRANSFER" == download ]] && (( download_status == 0 )); then
    echo "Diagnostic archive: $RESULT" >&2
  elif [[ "$TRANSFER" == download ]]; then
    echo 'Diagnostic archive could not be downloaded.' >&2
  fi
  exit "$exec_status"
fi
if (( download_status != 0 )); then exit "$download_status"; fi
if [[ "$TRANSFER" == download ]]; then
  echo "Downloaded: $RESULT"
else
  echo 'Run complete; no archive downloaded. Remote files will be discarded when the session stops.'
fi
