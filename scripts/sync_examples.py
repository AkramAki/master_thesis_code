"""Copy up to 15 raw FITS/UV pairs (30 files) from the configured SSH sources."""

import os
from pathlib import Path
import shlex
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radio_diffusion.paths import get_path, load_env


def main() -> None:
    load_env()
    fits_source = os.environ["MOJAVE_REMOTE_FITS"]
    uv_source = os.environ["MOJAVE_REMOTE_UV"]
    host, remote_dir = fits_source.split(":", 1)

    # A fixed, sorted selection keeps repeated runs on the same small sample.
    command = (
        f"find {shlex.quote(remote_dir)} -maxdepth 1 -type f "
        "-name '*.icn.fits.gz' -printf '%f\\n' | sort | head -n 15"
    )
    result = subprocess.run(
        ["ssh", host, command], check=True, capture_output=True, text=True,
    )
    fits_names = result.stdout.splitlines()[:15]
    if not fits_names:
        raise SystemExit(f"No raw FITS files found at {fits_source}")
    uv_names = [name.removesuffix(".icn.fits.gz") + ".uvf" for name in fits_names]

    for source, path_name, names in (
        (fits_source, "MOJAVE_FITS_DIR", fits_names),
        (uv_source, "MOJAVE_UV_DIR", uv_names),
    ):
        destination = get_path(path_name)
        destination.mkdir(parents=True, exist_ok=True)
        # Missing UV counterparts are skipped; existing local files are kept.
        subprocess.run(
            [
                "rsync", "-av", "--ignore-missing-args", "--files-from=-",
                "--", source.rstrip("/") + "/", str(destination) + "/",
            ],
            input="\n".join(names) + "\n", text=True, check=True,
        )


if __name__ == "__main__":
    main()
