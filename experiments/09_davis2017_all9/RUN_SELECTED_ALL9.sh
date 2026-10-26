#!/usr/bin/env bash
# DAVIS all-nine-method runner: always use the existing conda environment kpd_cv.
# Usage: bash experiments/09_davis2017_all9/RUN_SELECTED_ALL9.sh --videos blackswan [--jobs 2]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${KPD_CV_ENV_NAME:-kpd_cv}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    # Find the Conda base even in noninteractive server shells.
    CONDA_BASE=""
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
    if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        for candidate in "${CONDA_PREFIX:-}" "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/miniforge3" "${HOME}/mambaforge" "/opt/conda"; do
            if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
                CONDA_BASE="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        echo '[ERROR] Cannot locate Conda. First run: conda activate kpd_cv' >&2
        echo '[HINT] Or set KPD_CV_ENV_NAME / ensure conda is on PATH in this shell.' >&2
        exit 2
    fi
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME" || { echo "[ERROR] Existing conda env '$ENV_NAME' not found" >&2; exit 2; }
fi

# Enforce the environment's own interpreter; ignore a stale PYTHON env var.
if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo '[ERROR] CONDA_PREFIX/bin/python not found; activate kpd_cv again.' >&2
    exit 2
fi
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

echo "[ENV] conda=$CONDA_DEFAULT_ENV  python=$PYTHON_BIN"
cd "$ROOT"

if [[ "${1:-}" == '--check-env' ]]; then
    shift
    exec "$PYTHON_BIN" experiments/09_davis2017_all9/check_server.py --strict-nine "$@"
fi
exec "$PYTHON_BIN" experiments/09_davis2017_all9/run_selected_all9.py "$@"
