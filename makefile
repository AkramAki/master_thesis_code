# Usage: make create-env REPOS_DIR="$HOME/Documents/GitRepos"
# REPOS_DIR has no default: callers must choose where the repositories live.
REPOS_DIR ?=
ENV_NAME ?= radio-dev
PYTHON_VERSION ?= 3.12
DELETE_REPOS ?= 0
FORCE ?= 0

export REPOS_DIR ENV_NAME PYTHON_VERSION DELETE_REPOS FORCE

.PHONY: create-env delete-env help

create-env:
	bash environment/create.sh

delete-env:
	bash environment/delete.sh

help:
	@printf '%s\n' \
	  'make create-env REPOS_DIR="/path/to/repos" [ENV_NAME=radio-dev] [PYTHON_VERSION=3.12]' \
	  '  Optional: PYTHON_VERSION=3.12 can be overridden if the dependency stack supports it.' \
	  'make delete-env [ENV_NAME=radio-dev]' \
	  'make delete-env REPOS_DIR="/path/to/repos" DELETE_REPOS=1 [ENV_NAME=radio-dev]' \
	  'make delete-env REPOS_DIR="/path/to/repos" DELETE_REPOS=1 FORCE=1 [ENV_NAME=radio-dev]' \
	  'create-env also installs JupyterLab, catppuccin-jupyterlab, and nbstripout.' \
	  'DELETE_REPOS=1 also removes the four known checkouts when they are clean.' \
	  'FORCE=1 permits removing those checkouts when they contain local changes.' \
	  'Setup activation happens inside Make; activate the environment in your terminal afterward.'
