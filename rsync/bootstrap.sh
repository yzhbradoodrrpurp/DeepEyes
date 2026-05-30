#!/usr/bin/env bash
set -euo pipefail

# Direction: local -> remote only.
#
# This script uploads the repository to the remote code directory configured in
# .ssh_server.env. Prefer rsync/upload.sh for normal incremental work.
# SSH settings are read from the local-only env file:
#   ${REPO_ROOT}/.ssh_server.env
#
# It does not use --delete, so extra files already present on the server
# are not removed.
# It also uses --ignore-existing, so files already present in the remote
# target directory are not overwritten by local files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=rsync/lib/remote_env.sh
source "${SCRIPT_DIR}/lib/remote_env.sh"
require_remote_env "${REPO_ROOT}"

remote_mkdir "${SSH_REMOTE_CODE_DIR}"

run_with_password_prompt \
  rsync -az --progress \
  --ignore-existing \
  --exclude-from "${SCRIPT_DIR}/ignore" \
  -e "${RSYNC_SSH_CMD}" \
  "${REPO_ROOT}/" \
  "${REMOTE_TARGET}:${SSH_REMOTE_CODE_DIR}/"
