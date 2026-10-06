#!/usr/bin/env bash
set -e -o pipefail

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ "$(uname -s)" == Linux ]] ||
    fail 'Unsupported platform: this setup only supports Linux.'

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Validate before activating anything, creating directories, or installing packages.
[[ -n "${REPOS_DIR//[[:space:]]/}" ]] ||
    fail 'REPOS_DIR is required. Usage: make create-env REPOS_DIR="/path/to/repos"'
ENV_NAME="${ENV_NAME:-radio-dev}"
WITH_CUDA="${WITH_CUDA:-1}"
[[ "$ENV_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "$ENV_NAME" != base ]] ||
    fail 'ENV_NAME must be a simple environment name other than base.'
[[ "$WITH_CUDA" == 0 || "$WITH_CUDA" == 1 ]] ||
    fail 'WITH_CUDA must be 0 or 1.'

command -v mamba >/dev/null || fail 'mamba must be installed and available on PATH.'
command -v git >/dev/null || fail 'git must be installed and available on PATH.'

# Make runs a noninteractive shell, so initialize activation explicitly.
# Keep base active throughout setup; every install below names its target prefix.
mamba_hook="$(mamba shell hook --shell bash)"
eval "$mamba_hook"
mamba activate base

mkdir -p -- "$REPOS_DIR"
REPOS_DIR="$(cd -- "$REPOS_DIR" && pwd -P)"
repos=(radiotools pyvisgrid pyvisgen radiosim)

# Reject unrelated directories, including ordinary folders inside another Git repo.
for repo in "${repos[@]}"; do
    repo_dir="$REPOS_DIR/$repo"
    if [[ -e "$repo_dir" || -L "$repo_dir" ]]; then
        repo_root="$(git -C "$repo_dir" rev-parse --show-toplevel)" ||
            fail "$repo_dir exists but is not a Git checkout."
        [[ "$(cd -- "$repo_dir" && pwd -P)" == "$(cd -- "$repo_root" && pwd -P)" ]] ||
            fail "$repo_dir is not the root of a Git checkout."
        [[ -f "$repo_dir/pyproject.toml" ]] || fail "$repo_dir has no pyproject.toml."
    fi
done

environment_exists() {
    mamba list --name "$ENV_NAME" >/dev/null 2>&1
}

environment_prefix() {
    mamba run --name "$ENV_NAME" python -c \
        'import os, sys; print(os.path.realpath(sys.prefix))'
}

environment_created=0
if ! environment_exists; then
    mamba create --yes --name "$ENV_NAME" --channel conda-forge python uv
    environment_created=1
fi

# Use radio-dev's Python to resolve its own prefix. An existing environment
# without Python is incomplete, so repair its core packages first.
if ! env_prefix="$(environment_prefix)"; then
    mamba install --yes --name "$ENV_NAME" --channel conda-forge python uv
    env_prefix="$(environment_prefix)" ||
        fail "Could not locate Python in environment $ENV_NAME."
fi
env_python="$env_prefix/bin/python"

validate_environment() {
    [[ -x "$env_python" ]] || return 1
    for repo in "${repos[@]}"; do
        [[ -f "$REPOS_DIR/$repo/pyproject.toml" ]] || return 1
        git -C "$REPOS_DIR/$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
            return 1
    done
    mamba run --prefix "$env_prefix" uv pip check --python "$env_python" >/dev/null ||
        return 1
    mamba run --prefix "$env_prefix" "$env_python" - \
        "$REPOS_DIR" "$WITH_CUDA" "$PROJECT_ROOT" <<'PY'
import importlib
import json
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlsplit

repos_dir = Path(sys.argv[1])
project_root = Path(sys.argv[3])
for module, distribution, source_dir in (
    ("radiotools", "radionets-radiotools", repos_dir / "radiotools"),
    ("pyvisgrid", "pyvisgrid", repos_dir / "pyvisgrid"),
    ("pyvisgen", "pyvisgen", repos_dir / "pyvisgen"),
    ("radiosim", "radiosim", repos_dir / "radiosim"),
    ("radio_diffusion", "radio-diffusion", project_root),
):
    installed = metadata.distribution(distribution)
    source = json.loads(installed.read_text("direct_url.json") or "{}")
    url = urlsplit(source.get("url", ""))
    if (not source.get("dir_info", {}).get("editable")
            or url.scheme != "file"
            or Path(unquote(url.path)).resolve() != source_dir.resolve()):
        sys.exit(f"{distribution} is not editable from {source_dir}")
    importlib.import_module(module)
if sys.argv[2] == "1":
    importlib.import_module("cufinufft")
PY
}

if validate_environment >/dev/null 2>&1; then
    printf 'Environment %s is already complete; nothing to install.\n' "$ENV_NAME"
    printf 'Activate it in your terminal with:\n  mamba activate %s\n' "$ENV_NAME"
    exit 0
fi

printf 'Creating or repairing environment: %s\n' "$env_prefix"
if [[ "$environment_created" == 0 ]]; then
    mamba install --yes --name "$ENV_NAME" --channel conda-forge python uv
fi
env_python="$env_prefix/bin/python"
[[ -x "$env_python" ]] || fail "Python is missing from $env_prefix."

for repo in "${repos[@]}"; do
    repo_dir="$REPOS_DIR/$repo"
    if [[ ! -e "$repo_dir" ]]; then
        git clone "https://github.com/radionets-project/$repo.git" "$repo_dir"
    fi
    printf 'Installing %s with dev dependencies\n' "$repo"
    (
        cd -- "$repo_dir"
        # --python prevents an inherited VIRTUAL_ENV or local .venv taking precedence.
        mamba run --prefix "$env_prefix" uv pip install \
            --python "$env_python" --group dev -e .
    )
done

printf 'Installing radio-diffusion in editable mode\n'
(
    cd -- "$PROJECT_ROOT"
    mamba run --prefix "$env_prefix" uv pip install \
        --python "$env_python" -e .
)

if [[ "$WITH_CUDA" == 1 ]]; then
    mamba install --yes --prefix "$env_prefix" --channel conda-forge cufinufft
else
    printf 'Skipping cuFINUFFT (WITH_CUDA=0).\n'
fi

validate_environment || fail "Environment $ENV_NAME failed its final validation."

printf '\nEnvironment %s is ready. Activate it in your terminal with:\n  mamba activate %s\n' \
    "$ENV_NAME" "$ENV_NAME"
