"""Experiment settings, their validation, and TOML loading."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import tomllib
from typing import Any, Literal
import warnings


def _check_type(value: Any, expected: type | tuple[type, ...], name: str) -> None:
    """Check a value's type without accepting booleans as integers."""
    if isinstance(value, bool) and expected is not bool:
        raise ValueError(f"{name} has an invalid type.")
    if not isinstance(value, expected):
        raise ValueError(f"{name} has an invalid type.")


def _check_int(value: Any, name: str, minimum: int = 0) -> None:
    """Check an integer and its minimum allowed value."""
    _check_type(value, int, name)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


def _check_number(value: Any, name: str, minimum: float = 0.0) -> None:
    """Check a finite number and its minimum allowed value."""
    _check_type(value, (int, float), name)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")


def _check_text(value: Any, name: str) -> None:
    """Check a nonempty string."""
    _check_type(value, str, name)
    if not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _check_int_list(value: Any, name: str, minimum: int = 0) -> None:
    """Check a list of integers and their minimum allowed value."""
    _check_type(value, list, name)
    for item in value:
        _check_int(item, name, minimum)


def _build_settings(settings_class: type, values: Any, section: str) -> Any:
    """Construct a settings class from a TOML table."""
    _check_type(values, dict, section)
    try:
        return settings_class(**values)
    except TypeError as exc:
        raise ValueError(f"{section}: {exc}") from exc


@dataclass
class RunSettings:
    """Run name, random seed, and the optional legacy output path."""

    name: str
    seed: int
    output_dir: str | None = None

    def __post_init__(self) -> None:
        _check_text(self.name, "run.name")
        _check_int(self.seed, "run.seed")
        if self.output_dir is not None:
            _check_text(self.output_dir, "run.output_dir")


@dataclass
class TrackingSettings:
    """Optional MLflow tracking settings."""

    enabled: bool = False
    study: str | None = None

    def __post_init__(self) -> None:
        _check_type(self.enabled, bool, "tracking.enabled")
        if self.study is not None:
            _check_text(self.study, "tracking.study")


@dataclass
class TrainerSettings:
    """Run length, accelerator, and device settings."""

    max_epochs: int = 100
    accelerator: str = "auto"
    devices: int | str | list[int] = "auto"
    log_every_n_steps: int = 50
    check_val_every_n_epoch: int = 1

    def __post_init__(self) -> None:
        _check_int(self.max_epochs, "trainer.max_epochs", -1)
        _check_text(self.accelerator, "trainer.accelerator")
        _check_int(self.log_every_n_steps, "trainer.log_every_n_steps", 1)
        _check_int(self.check_val_every_n_epoch, "trainer.check_val_every_n_epoch", 1)
        self._check_devices()

    def _check_devices(self) -> None:
        if isinstance(self.devices, str):
            if self.devices not in ("auto", "-1"):
                indices = self.devices.split(",")
                if any(not index.strip().isdigit() for index in indices):
                    raise ValueError("trainer.devices must contain GPU indices or 'auto'.")
                if len(set(int(index) for index in indices)) != len(indices):
                    raise ValueError("trainer.devices must contain unique GPU indices.")
        elif isinstance(self.devices, list):
            _check_int_list(self.devices, "trainer.devices")
            if not self.devices or len(set(self.devices)) != len(self.devices):
                raise ValueError("trainer.devices must contain unique GPU indices.")
        else:
            _check_int(self.devices, "trainer.devices", -1)
            if self.devices == 0:
                raise ValueError("trainer.devices must be positive or -1.")


@dataclass
class NormalizationSettings:
    """Transform intensities, then rescale their minimum/maximum to zero/one.

    Shared bounds use the dataset-wide cropped-pixel minimum and maximum from
    the saved summary. Per-image bounds use each crop's saved extrema instead.
    Both bounds receive the same intensity transformation as the image before
    rescaling. Statistics remain in physical units in preprocessing metadata.
    log1p applies log(1 + x/log_scale); signed_log1p applies
    sign(x) * log(1 + abs(x)/log_scale). Disabled normalization is the identity.
    """

    enabled: bool = False
    transform: Literal["linear", "log1p", "signed_log1p"] = "linear"
    bounds: Literal["shared", "per_image"] = "shared"
    log_scale: float = 1.0

    def __post_init__(self) -> None:
        _check_type(self.enabled, bool, "data.normalization.enabled")
        if self.transform not in ("linear", "log1p", "signed_log1p"):
            raise ValueError("data.normalization.transform must be linear, log1p or signed_log1p.")
        if self.bounds not in ("shared", "per_image"):
            raise ValueError("data.normalization.bounds must be shared or per_image.")
        _check_number(self.log_scale, "data.normalization.log_scale")
        if self.log_scale == 0:
            raise ValueError("data.normalization.log_scale must be positive.")


@dataclass
class DataSettings:
    """MOJAVE dataset location, dimensions, and batch settings."""

    type: Literal["mojave"]
    batch_size: int = 16
    num_workers: int = 0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int | None = None
    image_size: int | None = None
    data_dir: str | None = None
    remote_source: str | None = None
    variant: str = "mojave_unspecified"
    normalization: NormalizationSettings = field(default_factory=NormalizationSettings)

    def __post_init__(self) -> None:
        if self.type != "mojave":
            raise ValueError("data.type must be 'mojave'.")
        _check_int(self.batch_size, "data.batch_size", 1)
        _check_int(self.num_workers, "data.num_workers")
        for name in ("val_fraction", "test_fraction"):
            _check_number(getattr(self, name), f"data.{name}")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("data.val_fraction + data.test_fraction must be below 1.")
        if self.seed is not None:
            _check_int(self.seed, "data.seed")
        if self.image_size is not None:
            _check_int(self.image_size, "data.image_size", 1)
        for name in ("data_dir", "remote_source"):
            if getattr(self, name) is not None:
                _check_text(getattr(self, name), f"data.{name}")
        _check_text(self.variant, "data.variant")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.variant):
            raise ValueError("data.variant must contain letters, digits, '_' or '-'.")
        if isinstance(self.normalization, dict):
            self.normalization = _build_settings(
                NormalizationSettings, self.normalization, "data.normalization"
            )
        _check_type(self.normalization, NormalizationSettings, "data.normalization")


@dataclass
class ConditioningSettings:
    """Mask encoding for model conditioning."""

    use_validity_masks: bool = False

    def __post_init__(self) -> None:
        _check_type(
            self.use_validity_masks, bool, "model.conditioning_settings.use_validity_masks"
        )


@dataclass
class ModelSettings:
    """Image channels and metadata settings shared by the supported models."""

    type: str
    in_channels: int = 1
    out_channels: int | None = None
    conditioning_params: list[str] = field(default_factory=list)
    conditioning_settings: ConditioningSettings = field(default_factory=ConditioningSettings)

    def __post_init__(self) -> None:
        _check_int(self.in_channels, "model.in_channels", 1)
        if self.out_channels is None:
            self.out_channels = self.in_channels
        _check_int(self.out_channels, "model.out_channels", 1)
        _check_type(self.conditioning_params, list, "model.conditioning_params")
        for name in self.conditioning_params:
            _check_text(name, "model.conditioning_params")
        if len(set(self.conditioning_params)) != len(self.conditioning_params):
            raise ValueError("model.conditioning_params must contain unique field names.")
        if isinstance(self.conditioning_settings, dict):
            self.conditioning_settings = _build_settings(
                ConditioningSettings, self.conditioning_settings, "model.conditioning_settings"
            )
        _check_type(self.conditioning_settings, ConditioningSettings, "model.conditioning_settings")
        if self.type != "conditioned_dit" and (
            self.conditioning_params or self.conditioning_settings.use_validity_masks
        ):
            warnings.warn(
                f"Model type {self.type!r} ignores conditioning settings.", stacklevel=2
            )


@dataclass
class UNetSettings(ModelSettings):
    """UNet widths, blocks, time embedding, and attention settings."""

    type: Literal["unet"] = "unet"
    channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    blocks_per_level: int | list[int] = 2
    time_embedding_dim: int = 128
    num_groups: int = 8
    attention_levels: list[int] = field(default_factory=list)
    attention_heads: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type != "unet":
            raise ValueError("UNet settings require model.type = 'unet'.")
        _check_int_list(self.channels, "model.channels", 1)
        if len(self.channels) < 2:
            raise ValueError("model.channels must contain at least two channel sizes.")
        for name in ("time_embedding_dim", "num_groups", "attention_heads"):
            _check_int(getattr(self, name), f"model.{name}", 1)
        if self.time_embedding_dim < 2 or self.time_embedding_dim % 2:
            raise ValueError("model.time_embedding_dim must be even and at least 2.")
        if any(channels % self.num_groups for channels in self.channels):
            raise ValueError("model.channels must be divisible by model.num_groups.")
        self._check_blocks()
        self._check_attention()

    def _check_blocks(self) -> None:
        if isinstance(self.blocks_per_level, list):
            _check_int_list(self.blocks_per_level, "model.blocks_per_level")
            counts = self.blocks_per_level
        else:
            _check_int(self.blocks_per_level, "model.blocks_per_level")
            counts = [self.blocks_per_level] * len(self.channels)
        if len(counts) != len(self.channels):
            raise ValueError("model.blocks_per_level must have one value per level.")
        if any(count < 1 for count in counts[:-1]):
            raise ValueError("Every UNet level except the deepest needs at least one block.")

    def _check_attention(self) -> None:
        _check_int_list(self.attention_levels, "model.attention_levels")
        if len(set(self.attention_levels)) != len(self.attention_levels):
            raise ValueError("model.attention_levels must contain unique levels.")
        if any(level >= len(self.channels) for level in self.attention_levels):
            raise ValueError("model.attention_levels contains an invalid level index.")
        if any(self.channels[level] % self.attention_heads for level in self.attention_levels):
            raise ValueError("Attention channels must be divisible by model.attention_heads.")


@dataclass
class DiTSettings(ModelSettings):
    """DiT patch size, transformer width, depth, and attention settings."""

    type: Literal["dit", "conditioned_dit"] = "dit"
    patch_size: int = 8
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type not in ("dit", "conditioned_dit"):
            raise ValueError("DiT settings require model.type = 'dit' or 'conditioned_dit'.")
        for name in ("patch_size", "hidden_size", "depth", "num_heads"):
            _check_int(getattr(self, name), f"model.{name}", 1)
        if self.hidden_size % 4:
            raise ValueError("model.hidden_size must be a multiple of 4.")
        if self.hidden_size % self.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads.")
        _check_number(self.mlp_ratio, "model.mlp_ratio")
        if self.mlp_ratio <= 0 or not math.isfinite(self.hidden_size * self.mlp_ratio):
            raise ValueError("model.mlp_ratio must produce a finite positive MLP width.")
        if int(self.hidden_size * self.mlp_ratio) < 1:
            raise ValueError("model.mlp_ratio must produce at least one MLP feature.")


@dataclass
class ConditionedDiTSettings(DiTSettings):
    """DiT settings with metadata conditioning enabled."""

    type: Literal["conditioned_dit"] = "conditioned_dit"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type != "conditioned_dit":
            raise ValueError("ConditionedDiT settings require model.type = 'conditioned_dit'.")


@dataclass
class DiffusionSettings:
    """Number of diffusion steps and the linear noise schedule."""

    num_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        _check_int(self.num_steps, "diffusion.num_steps", 1)
        _check_number(self.beta_start, "diffusion.beta_start")
        _check_number(self.beta_end, "diffusion.beta_end")
        if not 0 < self.beta_start <= self.beta_end < 1:
            raise ValueError("The schedule requires 0 < beta_start <= beta_end < 1.")


@dataclass
class LossSettings:
    """Noise-prediction loss names and their weights."""

    loss: list[str] = field(default_factory=lambda: ["MSELoss"])
    weights: list[float] = field(default_factory=lambda: [1.0])

    def __post_init__(self) -> None:
        _check_type(self.loss, list, "loss.loss")
        _check_type(self.weights, list, "loss.weights")
        if not self.loss or len(self.loss) != len(self.weights):
            raise ValueError("loss.weights must have one weight per loss in loss.loss.")
        supported = ("MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss")
        for name in self.loss:
            _check_text(name, "loss.loss")
            if name not in supported:
                raise ValueError(f"Unsupported noise-prediction loss: {name!r}.")
        if len(set(self.loss)) != len(self.loss):
            raise ValueError("loss.loss must not contain duplicate loss names.")
        for weight in self.weights:
            _check_number(weight, "loss.weights")
        try:
            total = math.fsum(self.weights)
        except OverflowError:
            total = math.inf
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"loss.weights must sum to 1; got {total}.")


@dataclass
class OptimizerSettings:
    """Adam or AdamW and their optimization settings."""

    type: Literal["Adam", "AdamW"]
    learning_rate: float
    weight_decay: float | None = None
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    amsgrad: bool = False

    def __post_init__(self) -> None:
        if self.type not in ("Adam", "AdamW"):
            raise ValueError("optimizer.type must be 'Adam' or 'AdamW'.")
        _check_number(self.learning_rate, "optimizer.learning_rate")
        if self.learning_rate == 0:
            raise ValueError("optimizer.learning_rate must be positive.")
        if self.weight_decay is None:
            self.weight_decay = 0.01 if self.type == "AdamW" else 0.0
        _check_number(self.weight_decay, "optimizer.weight_decay")
        _check_number(self.eps, "optimizer.eps")
        _check_type(self.amsgrad, bool, "optimizer.amsgrad")
        _check_type(self.betas, list, "optimizer.betas")
        if len(self.betas) != 2:
            raise ValueError("optimizer.betas must contain two numbers.")
        for beta in self.betas:
            _check_number(beta, "optimizer.betas")
            if beta >= 1:
                raise ValueError("optimizer.betas must be below 1.")


@dataclass
class SampleConditioningSettings:
    """Automatic conditioning or manual metadata values for sample images."""

    mode: Literal["manual", "auto"] = "manual"
    seed: int | None = None
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("manual", "auto"):
            raise ValueError("samples.conditioning.mode must be 'manual' or 'auto'.")
        if self.seed is not None:
            _check_int(self.seed, "samples.conditioning.seed")
        _check_type(self.values, dict, "samples.conditioning")
        if self.mode == "auto" and self.values:
            raise ValueError("Automatic sample conditioning does not accept manual values.")
        for name, value in self.values.items():
            _check_text(name, "samples.conditioning field")
            for item in value if isinstance(value, list) else [value]:
                if not isinstance(item, bool):
                    _check_number(item, f"samples.conditioning.{name}", -math.inf)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SampleConditioningSettings":
        """Separate conditioning mode and seed from the metadata values."""
        settings = dict(values)
        mode = settings.pop("mode", "manual")
        seed = settings.pop("seed", None)
        return cls(mode=mode, seed=seed, values=settings)


@dataclass
class SamplesSettings:
    """Sample image dimensions, frequency, and output settings."""

    enabled: bool = True
    num_samples: int | None = None
    shape: list[int] | None = None
    every_n_epochs: int = 10
    seed: int | None = None
    nrow: int = 4
    save_final_grid: bool = True
    save_denoising_grids: bool = True
    conditioning: SampleConditioningSettings | None = None

    def __post_init__(self) -> None:
        for name in ("enabled", "save_final_grid", "save_denoising_grids"):
            _check_type(getattr(self, name), bool, f"samples.{name}")
        for name in ("every_n_epochs", "nrow"):
            _check_int(getattr(self, name), f"samples.{name}", 1)
        if self.num_samples is not None:
            _check_int(self.num_samples, "samples.num_samples", 1)
        if self.seed is not None:
            _check_int(self.seed, "samples.seed")
        if self.shape is not None:
            _check_int_list(self.shape, "samples.shape", 1)
            if len(self.shape) != 4:
                raise ValueError("samples.shape must contain [batch, channels, height, width].")
        if isinstance(self.conditioning, dict):
            self.conditioning = SampleConditioningSettings.from_dict(self.conditioning)
        if self.conditioning is not None:
            _check_type(self.conditioning, SampleConditioningSettings, "samples.conditioning")


@dataclass
class ExperimentConfig:
    """Validated experiment sections and the source TOML path."""

    run: RunSettings
    model: UNetSettings | DiTSettings | ConditionedDiTSettings
    data: DataSettings
    diffusion: DiffusionSettings
    optimizer: OptimizerSettings
    source_path: Path
    trainer: TrainerSettings = field(default_factory=TrainerSettings)
    loss: LossSettings = field(default_factory=LossSettings)
    samples: SamplesSettings = field(default_factory=SamplesSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)

    def __post_init__(self) -> None:
        section_types = {
            "run": RunSettings, "model": (UNetSettings, DiTSettings),
            "data": DataSettings, "diffusion": DiffusionSettings,
            "optimizer": OptimizerSettings, "trainer": TrainerSettings,
            "loss": LossSettings, "samples": SamplesSettings, "tracking": TrackingSettings,
        }
        for name, expected in section_types.items():
            _check_type(getattr(self, name), expected, name)
        _check_type(self.source_path, Path, "source_path")
        if self.data.seed is None:
            self.data.seed = self.run.seed
        if self.samples.seed is None:
            self.samples.seed = self.run.seed
        self._resolve_sample_shape()
        self._check_normalization_conditioning()
        self._check_sample_conditioning()

    @property
    def sample_shape(self) -> tuple[int, int, int, int]:
        """The resolved sample count, channels, height, and width."""
        return tuple(self.samples.shape)

    @property
    def conditioning_params(self) -> list[str]:
        """The metadata fields used by the configured model."""
        if self.model.type != "conditioned_dit":
            return []
        return list(self.model.conditioning_params)

    @property
    def conditioning_settings(self) -> dict[str, bool]:
        """The mask settings supported by the configured model."""
        if self.model.type != "conditioned_dit":
            return {}
        return {"use_validity_masks": self.model.conditioning_settings.use_validity_masks}

    def _resolve_sample_shape(self) -> None:
        shape = self.samples.shape or [4, self.model.in_channels, 256, 256]
        if shape[1] != self.model.in_channels:
            raise ValueError("samples.shape channels must match model.in_channels.")
        if self.model.out_channels != self.model.in_channels:
            raise ValueError("model.out_channels must match model.in_channels for noise prediction.")
        count = self.samples.num_samples if self.samples.num_samples is not None else shape[0]
        height, width = shape[2:]
        if self.data.image_size is not None:
            height = width = self.data.image_size
        divisor = (
            self.model.patch_size if isinstance(self.model, DiTSettings)
            else 2 ** (len(self.model.channels) - 1)
        )
        if height % divisor or width % divisor:
            raise ValueError(f"Image height and width must be divisible by {divisor} for this model.")
        self.samples.num_samples = count
        self.samples.shape = [count, self.model.in_channels, height, width]

    def _check_sample_conditioning(self) -> None:
        conditioning = self.samples.conditioning
        fields = self.conditioning_params
        if conditioning is not None and conditioning.seed is None:
            conditioning.seed = self.samples.seed
        if not fields or not self.samples.enabled or not (
            self.samples.save_final_grid or self.samples.save_denoising_grids
        ):
            return
        if conditioning is None:
            raise ValueError("samples.conditioning must select auto mode or supply manual values.")
        if conditioning.mode == "auto":
            return
        if set(conditioning.values) != set(fields):
            raise ValueError("Manual samples.conditioning must supply exactly the model's fields.")
        for name, value in conditioning.values.items():
            if isinstance(value, list) and len(value) != self.samples.num_samples:
                raise ValueError(f"samples.conditioning.{name} must have one value per sample.")

    def _check_normalization_conditioning(self) -> None:
        """Require the physical scale as conditioning for per-image generation."""
        settings = self.data.normalization
        if not settings.enabled or settings.bounds != "per_image":
            return
        fields = ("cropped_image_pixel_minimum", "cropped_image_pixel_maximum")
        if not set(fields).issubset(self.conditioning_params):
            raise ValueError(
                "Per-image normalization requires a conditioned model with "
                "cropped_image_pixel_minimum and cropped_image_pixel_maximum "
                "in model.conditioning_params."
            )
        conditioning = self.samples.conditioning
        if not self.samples.enabled or conditioning is None or conditioning.mode == "auto":
            return
        bounds = []
        for field in fields:
            values = conditioning.values.get(field)
            if values is None:
                return
            values = values if isinstance(values, list) else [values] * self.samples.num_samples
            for value in values:
                _check_number(value, f"samples.conditioning.{field}", -math.inf)
            bounds.append(values)
        if any(lower > upper for lower, upper in zip(*bounds)):
            raise ValueError("Sample conditioning requires cropped pixel minimum <= maximum.")

    @classmethod
    def from_dict(cls, values: dict[str, Any], source_path: Path) -> "ExperimentConfig":
        """Build the section settings from parsed TOML values."""
        _check_type(values, dict, "config")
        required = {"run", "model", "data", "diffusion", "optimizer"}
        optional = {"trainer", "loss", "samples", "tracking"}
        unknown = set(values) - required - optional
        if unknown:
            raise ValueError(f"Unsupported config sections: {', '.join(sorted(unknown))}.")
        missing = required - set(values)
        if missing:
            raise ValueError(f"Missing config sections: {', '.join(sorted(missing))}.")
        _check_type(values["model"], dict, "model")
        model_type = values["model"].get("type")
        _check_text(model_type, "model.type")
        models = {"unet": UNetSettings, "dit": DiTSettings, "conditioned_dit": ConditionedDiTSettings}
        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type!r}.")
        sections = {
            "run": RunSettings, "model": models[model_type], "data": DataSettings,
            "diffusion": DiffusionSettings, "optimizer": OptimizerSettings,
            "trainer": TrainerSettings, "loss": LossSettings,
            "samples": SamplesSettings, "tracking": TrackingSettings,
        }
        settings = {
            name: _build_settings(settings_class, values.get(name, {}), name)
            for name, settings_class in sections.items()
        }
        return cls(**settings, source_path=source_path)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a TOML file and return its validated experiment settings.

    path: The path to the experiment config.
    """
    source_path = Path(path).expanduser().resolve()
    with source_path.open("rb") as config_file:
        values = tomllib.load(config_file)
    return ExperimentConfig.from_dict(values, source_path)
