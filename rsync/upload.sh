#!/usr/bin/env bash
set -euo pipefail

# Direction: local -> remote only.
#
# Upload only explicitly listed files or directories, preserving their paths
# relative to the repository root.
#
# SSH settings are read from the local-only env file:
#   ${REPO_ROOT}/.ssh_server.env
#
# This script does not use --delete. It overwrites only the selected paths on
# the remote side, so use it only after confirming the exact upload scope.
#
# Usage:
#   rsync/upload.sh path/to/file [path/to/dir ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=rsync/lib/remote_env.sh
source "${SCRIPT_DIR}/lib/remote_env.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path/to/file [path/to/dir ...]" >&2
  exit 1
fi

require_remote_env "${REPO_ROOT}"

sources=()
for path in "$@"; do
  case "${path}" in
    /*)
      echo "Upload paths must be relative to the repo root: ${path}" >&2
      exit 1
      ;;
    *..*)
      echo "Upload paths must not contain '..': ${path}" >&2
      exit 1
      ;;
  esac
  if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
    echo "Path does not exist: ${path}" >&2
    exit 1
  fi
  sources+=("./${path}")
done

remote_mkdir "${SSH_REMOTE_CODE_DIR}"

(
  cd "${REPO_ROOT}"
  run_with_password_prompt \
    rsync -az --relative \
    --exclude-from "${SCRIPT_DIR}/ignore" \
    -e "${RSYNC_SSH_CMD}" \
    "${sources[@]}" \
    "${REMOTE_TARGET}:${SSH_REMOTE_CODE_DIR}/"
)
