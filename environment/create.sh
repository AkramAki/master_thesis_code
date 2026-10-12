#!/usr/bin/env bash
set -e -o pipefail

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ "$(uname -s)" == Linux ]] ||
    fail 'Unsupported platform: this setup only supports Linux.'

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

[[ -n "${REPOS_DIR//[[:space:]]/}" ]] ||
    fail 'REPOS_DIR is required. Usage: make create-env REPOS_DIR="/path/to/repos"'
ENV_NAME="${ENV_NAME:-radio-dev}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
[[ "$ENV_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "$ENV_NAME" != base ]] ||
    fail 'ENV_NAME must be a simple environment name other than base.'
[[ "$PYTHON_VERSION" =~ ^[0-9]+[.][0-9]+$ ]] ||
    fail 'PYTHON_VERSION must be a Python minor version like 3.12.'

command -v mamba >/dev/null || fail 'mamba must be installed and available on PATH.'
command -v git >/dev/null || fail 'git must be installed and available on PATH.'

packages=("python=$PYTHON_VERSION" uv nbstripout jupyterlab catppuccin-jupyterlab)

if mamba list --name "$ENV_NAME" >/dev/null 2>&1; then
    printf 'Updating environment: %s\n' "$ENV_NAME"
    mamba install --yes --name "$ENV_NAME" --channel conda-forge "${packages[@]}"
else
    printf 'Creating environment: %s\n' "$ENV_NAME"
    mamba create --yes --name "$ENV_NAME" --channel conda-forge "${packages[@]}"
fi

env_prefix="$(mamba run --name "$ENV_NAME" python -c 'import os, sys; print(os.path.realpath(sys.prefix))')"
env_python="$env_prefix/bin/python"

mkdir -p -- "$REPOS_DIR"
REPOS_DIR="$(cd -- "$REPOS_DIR" && pwd -P)"

repos=(radiotools pyvisgrid pyvisgen radiosim)
for repo in "${repos[@]}"; do
    repo_dir="$REPOS_DIR/$repo"
    if [[ ! -d "$repo_dir" ]]; then
        git clone "https://github.com/radionets-project/$repo.git" "$repo_dir"
    fi

    printf 'Installing %s\n' "$repo"
    (
        cd -- "$repo_dir"
        mamba run --prefix "$env_prefix" uv pip install \
            --python "$env_python" --group dev -e .
    )
done

printf 'Installing radio-diffusion\n'
(
    cd -- "$PROJECT_ROOT"
    mamba run --prefix "$env_prefix" uv pip install \
        --python "$env_python" -e .
)

printf 'Installing nbstripout Git filter\n'
(
    cd -- "$PROJECT_ROOT"
    mamba run --prefix "$env_prefix" nbstripout --install --attributes .gitattributes
)

printf '\nEnvironment %s is ready. Activate it in your terminal with:\n  mamba activate %s\n' \
    "$ENV_NAME" "$ENV_NAME"
