"""Read machine-specific paths from the repository's .env file.

Settings use KEY=value lines; values containing spaces must be quoted.
Existing environment variables take precedence over .env values.
"""

import os
from pathlib import Path
import shlex


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load .env settings without depending on the current working directory."""
    env_file = REPOSITORY_ROOT / ".env"
    if not env_file.is_file():
        return

    for line_number, line in enumerate(env_file.read_text().splitlines(), start=1):
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        key, separator, value = parts[0].partition("=")
        if len(parts) != 1 or not separator or not key.isidentifier():
            raise ValueError(f"Expected KEY=value in {env_file}:{line_number}")
        os.environ.setdefault(key, value)


def get_path(name: str) -> Path:
    """Return a configured path, resolving relative paths at repository root."""
    load_env()
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} in {REPOSITORY_ROOT / '.env'} or the environment.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def get_mojave_data_dir(image_size: int) -> Path:
    """Return the processed MOJAVE directory for a positive image size."""
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
        raise ValueError("image_size must be a positive integer.")
    return get_path("MOJAVE_DATA_ROOT") / f"Mojave_I_{image_size}"


def get_preprocess_cpu_cores() -> int:
    """Return the configured preprocessing worker count, defaulting to one."""
    load_env()
    value = os.environ.get("MOJAVE_PREPROCESS_CPUS", "1")
    try:
        cpu_cores = int(value)
    except ValueError:
        raise ValueError("MOJAVE_PREPROCESS_CPUS must be a positive integer.") from None
    if cpu_cores < 1:
        raise ValueError("MOJAVE_PREPROCESS_CPUS must be a positive integer.")
    return cpu_cores
