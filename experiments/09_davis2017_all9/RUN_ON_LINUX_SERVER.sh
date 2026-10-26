#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/RUN_SELECTED_ALL9.sh" "$@"
