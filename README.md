# Radio Diffusion

Master's thesis research code for developing diffusion models with PyTorch
Lightning for radio astronomy. The goal is to train on MOJAVE images stored in
FITS files. MNIST is the initial dataset for learning and developing the pipeline.

Data loading, neural networks, diffusion methods, and training are separate
components so each can change independently. The current source layout is:

```text
src/radio_diffusion/
├── experiment.py
├── review.py
├── data/
│   └── mnist.py
├── models/
│   └── unet.py
├── diffusion/
│   └── ddpm.py
└── training/
    └── module.py
```

`models/unet.py` contains the reusable noise-prediction network.
`diffusion/ddpm.py` contains the self-contained Gaussian forward and reverse
processes, configurable noise schedule, and sampling loop. The Lightning
training logic lives in `training/module.py`. New modules are added only when
they have a real implementation; the longer-term direction is described in
[the structure plan](docs/structure-plan.md).

Use [notebooks/](notebooks/) for interactive checks and exploration. Notebooks
call the reusable code in `src/radio_diffusion/` and run with the development
environment as their kernel.

`review.py` provides reusable helpers for saved-run discovery, reproducible
inference, and comparison plots. In
[review_saved_run.ipynb](notebooks/review_saved_run.ipynb), list the saved runs,
choose models A and B, and set the seed near the top. The notebook shows shared
noise, adjacent 4×4 generated-image grids, archived training samples and losses,
then reconstructions of validation images held out from both runs. Run it in
the training environment with the saved outputs and dataset available.

## Setup

The automated environment setup currently supports Linux only. Choose the
parent directory in which the related RadioNets repositories should live:

```bash
make create-env REPOS_DIR="$HOME/GitRepos"
mamba activate radio-dev
```

This also installs JupyterLab, `catppuccin-jupyterlab`, and `nbstripout`, then
configures the repository-local Git filter so notebook outputs are stripped
before commit.

Also inside the radiosim repo run
```bash
git switch fix_output_path  
```

The setup uses Python 3.12 by default. Use `PYTHON_VERSION=3.x` when the
dependency stack supports another Python minor version. Run `make help` for the
environment deletion options.

To install or update just this package and its dependencies in an existing
Python 3.11+ environment:

```bash
python -m pip install -e .
```
