# Radio Diffusion

Master's thesis research code for developing diffusion models with PyTorch
Lightning for radio astronomy. The goal is to train on MOJAVE images stored in
FITS files. MNIST is the initial dataset for learning and developing the pipeline.

Data loading, neural networks, diffusion, and training will be separate components
so datasets and experiment settings can be changed independently. The project
grows in small steps; the intended layout is described in
[the structure plan](docs/structure-plan.md).

Use [notebooks/](notebooks/) for interactive checks and exploration. Notebooks
call the reusable code in `src/radio_diffusion/` and run with the development
environment as their kernel.

## Setup

The automated environment setup currently supports Linux only. Choose the
parent directory in which the related RadioNets repositories should live:

```bash
make create-env REPOS_DIR="$HOME/GitRepos"
mamba activate radio-dev
```

This also installs `nbstripout` and configures the repository-local Git filter
so notebook outputs are stripped before commit.

Also inside the radiosim repo run
```bash
git switch fix_output_path  
```

Use `WITH_CUDA=0` when creating a CPU-only environment. Run `make help` for the
environment deletion options.

To install or update just this package and its dependencies in an existing
Python 3.11+ environment:

```bash
python -m pip install -e .
```
