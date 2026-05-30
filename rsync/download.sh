#!/usr/bin/env bash
set -euo pipefail

# Direction: remote -> local only.
#
# This script downloads remote results configured by REMOTE_RESULT_DIR or
# SSH_REMOTE_RESULTS_DIR.
# SSH settings are read from the local-only env file:
#   ${REPO_ROOT}/.ssh_server.env
#
# It does not use --delete, so local files in the target directory are not removed.
# It also uses --ignore-existing, so files already present in the local
# target directory are not overwritten by remote files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=rsync/lib/remote_env.sh
source "${SCRIPT_DIR}/lib/remote_env.sh"
require_remote_env "${REPO_ROOT}"

REMOTE_RESULT_DIR="${REMOTE_RESULT_DIR:-${SSH_REMOTE_RESULTS_DIR:-}}"
if [[ -z "${REMOTE_RESULT_DIR}" ]]; then
  echo "Set REMOTE_RESULT_DIR or SSH_REMOTE_RESULTS_DIR before downloading results." >&2
  exit 1
fi
require_absolute_remote_path "REMOTE_RESULT_DIR" "${REMOTE_RESULT_DIR}"

LOCAL_RESULT_DIR="${LOCAL_RESULT_DIR:-${REPO_ROOT}/results/downloaded}"

mkdir -p "${LOCAL_RESULT_DIR}"

run_with_password_prompt \
  rsync -az --progress \
  --ignore-existing \
  --exclude-from "${SCRIPT_DIR}/ignore" \
  -e "${RSYNC_SSH_CMD}" \
  "${REMOTE_TARGET}:${REMOTE_RESULT_DIR}/" \
  "${LOCAL_RESULT_DIR}/"
