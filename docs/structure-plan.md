# Project structure

The source tree separates data loading, network architectures, diffusion
methods, and training logic. Directories and modules are added only when their
first real implementation is needed.

```text
src/radio_diffusion/
├── data/
│   └── mnist.py
├── models/
│   └── unet.py
├── diffusion/
│   └── ddpm.py
└── training/
    └── module.py
```

The boundaries are intentionally direct:

- `data/` owns dataset loading and preprocessing.
- `models/` owns reusable PyTorch network architectures.
- `diffusion/` owns self-contained forward and reverse diffusion processes,
  their schedules, and sampling.
- `training/` owns Lightning training, validation, and optimizer logic.

This keeps the U-Net independent of DDPM so another diffusion method can reuse
it, and keeps diffusion operations independent of Lightning.

## Training architecture

When a reusable training entry point is needed, it should be conceptually
equivalent to:

```python
def train(config) -> TrainingResult:
    ...
```

One call performs one training run. It constructs the selected model and
DataModule, constructs the PyTorch Lightning `Trainer`, trains, and returns
metrics and checkpoint information. CLI and future Optuna code should call this
function instead of owning separate training logic.

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
