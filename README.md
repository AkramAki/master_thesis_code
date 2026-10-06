# Radio Diffusion

Research code for the master thesis. The repository starts with a small `src/`
layout and will grow as the implementation requires it.

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

The current repository contains only the environment helpers and the empty
`radio_diffusion` package.
