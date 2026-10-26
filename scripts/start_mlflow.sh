#!/usr/bin/env bash
set -euo pipefail

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'HELP'
Usage: bash scripts/start_mlflow.sh [--port PORT] [--data-dir PATH]

Run the MLflow server in the foreground for a tmux window. Stop with Ctrl+C.

Options:
  --port PORT      Local port (default: 5000).
  --data-dir PATH  Persistent storage (default: <repository>/outputs/mlflow).
                   Relative paths are resolved from the current directory.
  -h, --help       Show this help.

The server listens on 127.0.0.1. Use SSH port forwarding for remote access.
Restart with the same data directory to retain experiments and artifacts.
HELP
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
storage_dir="$project_root/outputs/mlflow"
server_port=5000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            [[ $# -ge 2 ]] || fail '--port requires a value.'
            server_port="$2"
            shift 2
            ;;
        --data-dir)
            [[ $# -ge 2 && -n "$2" ]] || fail '--data-dir requires a path.'
            storage_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1. Use --help for usage."
            ;;
    esac
done

[[ "$server_port" =~ ^[0-9]{1,5}$ ]] || fail '--port must be an integer from 1 to 65535.'
server_port=$((10#$server_port))
(( server_port >= 1 && server_port <= 65535 )) ||
    fail '--port must be an integer from 1 to 65535.'

command -v mlflow >/dev/null 2>&1 ||
    fail "MLflow is unavailable. Activate your environment and run: python -m pip install -e '.[tracking]' from the repository root."

mkdir -p -- "$storage_dir/artifacts"
cd -- "$storage_dir"
storage_dir="$(pwd -P)"

printf 'Starting MLflow at http://127.0.0.1:%s (Ctrl+C to stop).\n' "$server_port"
printf 'Database: %s/mlflow.db\nArtifacts: %s/artifacts\n' "$storage_dir" "$storage_dir"
printf 'In the training window: export MLFLOW_TRACKING_URI=http://127.0.0.1:%s\n' "$server_port"

exec mlflow server \
    --host 127.0.0.1 \
    --port "$server_port" \
    --workers 1 \
    --backend-store-uri "sqlite:///$storage_dir/mlflow.db" \
    --serve-artifacts \
    --default-artifact-root mlflow-artifacts:/ \
    --artifacts-destination "$storage_dir/artifacts"
