#!/usr/bin/env bash
# Multi-repo git wrapper. Configured repos are run in order; stop on first error.
# All arguments are passed through to git unchanged.
set -euo pipefail

# name|path  （~ 会在运行时展开）
REPOS=(
  "board|~/workdir/workboard/board"
  "notes|~/workdir/workboard/notes"
  "notes_browser|~/workdir/workboard/notes_browser"
  "transfer_hub|~/workdir/workboard/transfer_hub"
  "webui|~/workdir/workboard/webui"
)

usage() {
  cat <<'EOF' >&2
Usage: ./xgit.sh <git-args...>

All arguments are forwarded to git in each configured repository, e.g.:
  ./xgit.sh status -u
  ./xgit.sh checkout br_xxx
  ./xgit.sh checkout -b br_new_xxx
  ./xgit.sh add -A
  ./xgit.sh commit -m "msg"
  ./xgit.sh pull --rebase

Repositories are processed in configured order; stops on first failure.
EOF
}

# Print a red error line to stderr.
error() {
  printf '\033[1;31merror: %s\033[0m\n' "$*" >&2
}

expand_path() {
  local p="$1"
  # Quote the strip pattern: unquoted ~/ is tilde-expanded inside ${var#pat}.
  case "$p" in
    "~")
      printf '%s\n' "$HOME"
      ;;
    "~/"*)
      printf '%s\n' "$HOME/${p#"~/"}"
      ;;
    *)
      printf '%s\n' "$p"
      ;;
  esac
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

for entry in "${REPOS[@]}"; do
  name="${entry%%|*}"
  raw_path="${entry#*|}"
  path="$(expand_path "$raw_path")"

  if [[ ! -d "$path" ]]; then
    error "[${name}] path does not exist: ${path}"
    exit 1
  fi
  if ! git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    error "[${name}] is not a git repository: ${path}"
    exit 1
  fi

  # [${name}] bold green; path green
  printf '===== \033[1;32m[%s]\033[0m \033[32m%s\033[0m =====\n' "$name" "$path"

  set +e
  git -C "$path" "$@"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    # commit on a clean working tree: nothing to do — treat as success and continue
    if [[ "${1:-}" == "commit" ]] && [[ -z "$(git -C "$path" status --porcelain)" ]]; then
      continue
    fi
    error "[${name}] git $* failed (exit ${rc})"
    exit "$rc"
  fi
done