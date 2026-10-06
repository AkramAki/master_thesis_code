#!/usr/bin/env bash
set -e -o pipefail

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

ENV_NAME="${ENV_NAME:-radio-dev}"
DELETE_REPOS="${DELETE_REPOS:-0}"
FORCE="${FORCE:-0}"
[[ "$ENV_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "$ENV_NAME" != base ]] ||
    fail 'ENV_NAME must be a simple environment name other than base.'
[[ "$DELETE_REPOS" == 0 || "$DELETE_REPOS" == 1 ]] ||
    fail 'DELETE_REPOS must be 0 or 1.'
[[ "$FORCE" == 0 || "$FORCE" == 1 ]] || fail 'FORCE must be 0 or 1.'
command -v mamba >/dev/null || fail 'mamba must be installed and available on PATH.'

environment_exists() {
    mamba list --name "$ENV_NAME" >/dev/null 2>&1
}

repos=(radiotools pyvisgrid pyvisgen radiosim)
repos_to_delete=()

if [[ "$DELETE_REPOS" == 1 ]]; then
    [[ -n "${REPOS_DIR//[[:space:]]/}" ]] ||
        fail 'REPOS_DIR is required when DELETE_REPOS=1.'
    [[ -d "$REPOS_DIR" ]] || fail "Repository directory does not exist: $REPOS_DIR"
    command -v git >/dev/null || fail 'git must be installed and available on PATH.'
    REPOS_DIR="$(cd -- "$REPOS_DIR" && pwd -P)"

    # Validate every checkout before deleting the environment or any repository.
    for repo in "${repos[@]}"; do
        repo_dir="$REPOS_DIR/$repo"
        [[ -e "$repo_dir" || -L "$repo_dir" ]] || continue
        repo_root="$(git -C "$repo_dir" rev-parse --show-toplevel)" ||
            fail "$repo_dir exists but is not a Git checkout."
        [[ "$(cd -- "$repo_dir" && pwd -P)" == "$(cd -- "$repo_root" && pwd -P)" ]] ||
            fail "$repo_dir is not the root of a Git checkout."
        if [[ "$FORCE" == 0 && -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
            fail "$repo_dir has local changes. Commit or stash them, or use FORCE=1."
        fi
        repos_to_delete+=("$repo_dir")
    done
fi

if environment_exists; then
    printf 'Removing Mamba environment %s\n' "$ENV_NAME"
    mamba env remove --yes --name "$ENV_NAME"
    if environment_exists; then
        fail "Environment $ENV_NAME still exists after removal."
    fi
    printf 'Environment %s was deleted.\n' "$ENV_NAME"
else
    printf 'Environment %s does not exist; nothing to delete.\n' "$ENV_NAME"
fi

if [[ "$DELETE_REPOS" == 1 ]]; then
    for repo_dir in "${repos_to_delete[@]}"; do
        printf 'Removing repository %s\n' "$repo_dir"
        rm -rf -- "$repo_dir"
    done
else
    printf 'Repository checkouts were kept.\n'
fi
