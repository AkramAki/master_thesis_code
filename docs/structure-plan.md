# Project structure plan (generated with an LLM to maybe steer this repo)

This document records possible structure without requiring empty directories or
placeholder implementations before they are useful.

```text
configs/
├── data/
├── experiment/
├── model/
├── trainer/
└── tuning/

src/radio_diffusion/
├── data/
├── models/
│   └── ddpm/
│       ├── unet.py
│       ├── diffusion.py
│       └── module.py
├── training/
│   └── train.py
├── metrics/
├── callbacks/
└── utils/

scripts/
├── train.py
├── tune.py
├── evaluate.py
└── sample.py

notebooks/
data/
outputs/
tests/
```

Create each directory or module when its first real use appears.

## Training architecture

The reusable entry point should eventually be conceptually equivalent to:

```python
def train(config) -> TrainingResult:
    ...
```

One call performs one training run. It constructs the selected model and
DataModule, constructs the PyTorch Lightning `Trainer`, trains, and returns
metrics and checkpoint information. CLI and future Optuna code should call this
function instead of owning separate training logic.

The DDPM modules should have narrow responsibilities:

- `unet.py`: PyTorch network architecture.
- `diffusion.py`: noise schedules, forward/reverse diffusion, and sampling.
- `module.py`: Lightning training, validation, and optimizer logic.

## Configuration

Use plain YAML when configuration becomes necessary. Keep scientific settings
(model, dataset, optimizer, seed) separate from compute settings (devices,
precision, strategy, nodes). GPU count is a resource setting, not a model
hyperparameter.

Distributed batch sizing should be deliberate:

```text
global batch size = per-device batch size × devices × gradient accumulation
```

Future tuning should support both one multi-GPU trial at a time and multiple
single-GPU trials. No model implementation should assume that three GPUs are
always available.
