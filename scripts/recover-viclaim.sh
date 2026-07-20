#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-viclaim"
INPUT_CSV="${REPO_ROOT}/data/fact_opinion/viclaim_public/data/viclaim.csv"
OUTPUT_CSV="${REPO_ROOT}/data/fact_opinion/processed/viclaim_transcribed.csv"
CACHE_DIR="${REPO_ROOT}/data/fact_opinion/viclaim_recovery/cache"

if command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python 3.10 or newer is required." >&2
  exit 1
fi

"${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' || {
  echo "Python 3.10 or newer is required." >&2
  exit 1
}

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required by current yt-dlp YouTube extraction." >&2
  echo "Install it with 'brew install node' or your system package manager." >&2
  exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "ViClaim input CSV not found: ${INPUT_CSV}" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade pip wheel setuptools
"${PYTHON}" -m pip install --upgrade \
  "faster-whisper==1.2.1" \
  "pandas>=2.2" \
  "tqdm>=4.66" \
  "yt-dlp[default,curl-cffi]"

CLIP_ARGS=()
if [[ -n "${VICLAIM_CLIPS:-}" ]]; then
  read -r -a REQUESTED_CLIPS <<< "${VICLAIM_CLIPS}"
  CLIP_ARGS=(--clips "${REQUESTED_CLIPS[@]}")
fi

"${PYTHON}" "${REPO_ROOT}/scripts/recover_viclaim.py" \
  --input "${INPUT_CSV}" \
  --output "${OUTPUT_CSV}" \
  --cache-dir "${CACHE_DIR}" \
  "${CLIP_ARGS[@]}"

"${PYTHON}" "${REPO_ROOT}/back-end/ml.py"

echo
echo "ViClaim recovery and dataset rebuild complete."
echo "Recovered CSV: ${OUTPUT_CSV}"
echo "Dataset: ${REPO_ROOT}/data/fact_opinion/processed/fact_opinion.csv"
