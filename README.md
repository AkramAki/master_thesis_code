# Radio Diffusion

Diffusion-model experiments for radio astronomy with PyTorch Lightning.
MOJAVE FITS images are cropped into tensors for training; MNIST supports
initial exploration.

| Code | Purpose |
|---|---|
| `data/` | Preprocessing, datasets, and data loaders |
| `models/unet.py` | Time-conditioned U-Net architectures |
| `diffusion/ddpm.py` | Forward diffusion and reverse sampling |
| `training/` | Training steps, losses, callbacks, and tracking |
| `experiment.py` | Config loading, component construction, and checkpoint loading |
| `review.py` | Saved-run discovery, inference, and comparison plots |

These modules live under `src/radio_diffusion/`.
[Notebooks](notebooks/) provide interactive exploration and saved-run review.
The [structure plan](docs/structure-plan.md) describes the project boundaries.

## Setup

The environment setup supports Linux and defaults to Python 3.12:

```bash
make create-env REPOS_DIR="$HOME/GitRepos"
mamba activate radio-dev
```

This installs the related RadioNets repositories, JupyterLab,
`catppuccin-jupyterlab`, and the repository's `nbstripout` filter.
The radiosim checkout requires `git switch fix_output_path`.
`PYTHON_VERSION=3.x` overrides the Python version; `make help` lists environment
options.

Package installation in an existing Python 3.11+ environment:

```bash
python -m pip install -e .
```

## Training

The queue trains the TOML configs listed in `configs/train_queue.txt` in order:

```bash
python -m radio_diffusion.training.train_queue configs/train_queue.txt
```

Adding `--preflight` checks one synthetic training step per config on the
configured devices. It does not load the real dataset or create MLflow runs.
Each training run saves its config, checkpoints, metrics, and periodic samples
under its configured output directory.

## MLflow

The tracking dependency is optional:

```bash
python -m pip install -e '.[tracking]'
```

This script starts the MLflow server in the foreground, for example in a tmux
server window:

```bash
bash scripts/start_mlflow.sh
```

The server listens at `http://127.0.0.1:5000`. Its database and artifacts persist
under `outputs/mlflow/`. Ctrl+C stops the server. Optional settings:

```bash
bash scripts/start_mlflow.sh --port 5001 --data-dir /path/to/mlflow-storage
```

The training window sets the server address before starting the queue:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

This address is also the default. Training configs enable tracking with:

```toml
[tracking]
enabled = true
study = "architecture_comparison"
```

`study` is an optional tag. Omitting `[tracking]` or setting `enabled = false`
keeps CSV-only logging. Enabled tracking requires a reachable server.

Each training execution creates a run in its dataset experiment. MLflow records
config parameters, dataset metadata, loss curves, Git revision, and software
versions. CSV metrics remain available to
[review_saved_run.ipynb](notebooks/review_saved_run.ipynb). Final and intermediate
sample PNG grids are uploaded to MLflow; raw tensors and checkpoints remain
in the local run directory.

SSH forwarding exposes a remote server's dashboard on the local machine:

```bash
ssh -N -L 5000:127.0.0.1:5000 user@training-host
```

The dashboard then opens at `http://127.0.0.1:5000` in the local browser.
