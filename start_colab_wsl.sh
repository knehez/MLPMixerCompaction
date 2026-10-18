#!/usr/bin/env bash
# Google-hosted GPU workflow, matching ELM-cikk/scripts/run_elm_colab.sh.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/run_mixer_colab.sh" "$@"
