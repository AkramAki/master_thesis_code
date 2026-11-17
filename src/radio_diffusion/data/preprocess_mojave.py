"""Preprocess Mojave images and save crops with metadata."""

import argparse
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from radio_diffusion.paths import (
    get_mojave_data_dir,
    get_path,
    get_preprocess_cpu_cores,
    load_env,
)

load_env()

import torch
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

from radio_diffusion.utils.timing import timelogger
import warnings
from astropy.wcs import FITSFixedWarning

# Ignore FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

NOISE_METHOD = "four_corner"  # "header" or "four_corner".
JET_ORIENTATION_SETTINGS = {
    "inner_mas": 1.0,
    "outer_mas": 3.0,
    "sigma_cut": 3.0,
}
RIDGELINE_SETTINGS = {
    "radial_step_mas": 0.05,
    "max_radius_mas": None,
    "half_width_deg": 90.0,
    "angle_step_deg": 0.5,
    "sigma_cut": 8.0,
}


# Total VLBA flux: sum of all CLEAN-model component fluxes, including negatives.
# This follows Kovalev et al. (2005), Section 4, without a threshold or aperture.
# https://arxiv.org/pdf/astro-ph/0505536
# AIPS Memo 117, Section 6.1: FLUX contains component total flux in Jy.
# https://www.aips.nrao.edu/TEXT/PUBL/AIPSMEM117.PDF
def total_flux_jy(clean_components_hdu: fits.BinTableHDU) -> float:
    """Return the signed full CLEAN-model flux in Jy, before cropping."""
    if str(clean_components_hdu.header.get("EXTNAME", "")).strip().upper() != "AIPS CC":
        raise ValueError("Expected an AIPS CC table.")
    if "FLUX" not in clean_components_hdu.columns.names:
        raise ValueError("Missing FLUX column in CLEAN-component table.")
    unit = (clean_components_hdu.columns["FLUX"].unit or "").strip()
    if unit.upper() != "JY":
        raise ValueError(f"Expected CLEAN-component FLUX units of Jy; got {unit!r}.")
    if clean_components_hdu.data is None or len(clean_components_hdu.data) == 0:
        raise ValueError("Empty CLEAN-component table.")

    fluxes = np.asarray(clean_components_hdu.data["FLUX"], dtype=np.float64)
    if not np.isfinite(fluxes).all():
        raise ValueError("Non-finite CLEAN-component flux.")
    return float(fluxes.sum(dtype=np.float64))


# Peak brightness is the maximum pixel value in the supplied CLEAN image.
# Jy/beam is the image brightness convention used in Kovalev et al. (2005), Fig. 1.
# https://arxiv.org/pdf/astro-ph/0505536
def peak_brightness_jy_per_beam(image_hdu: fits.PrimaryHDU) -> float:
    """Return the maximum non-blank full-image brightness in Jy/beam, before cropping."""
    unit = str(image_hdu.header.get("BUNIT", "")).strip()
    if unit.replace(" ", "").upper() != "JY/BEAM":
        raise ValueError(f"Expected image units of Jy/beam; got {unit!r}")
    image = image_hdu.data
    if image is None:
        raise ValueError("Image HDU contains no pixel data.")
    if np.isinf(image).any() or not np.isfinite(image).any():
        raise ValueError("Image contains infinite values or has no finite pixels.")

    return float(np.nanmax(image))


# Pushkarev et al. (2017), Section 2.1: minimum RMS of four corner
# regions, each covering 1/16 of the image area.
# https://arxiv.org/pdf/1705.02888
# Numerical choices: RMS about zero, no clipping; quarter-side lengths
# rounded down. Blank pixels and unusable corner estimates are excluded.
def four_corner_rms_jy_per_beam(image_hdu: fits.PrimaryHDU) -> float:
    """Return minimum positive full-image corner RMS, or NaN if unavailable."""
    unit = str(image_hdu.header.get("BUNIT", "")).strip()
    if unit.replace(" ", "").upper() != "JY/BEAM":
        raise ValueError(f"Expected image units of Jy/beam; got {unit!r}")
    if image_hdu.data is None:
        raise ValueError("Image HDU contains no pixel data.")
    image = np.squeeze(image_hdu.data)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("Corner RMS estimation requires a 2D image of at least 8 x 8 pixels.")

    height, width = image.shape
    corner_height, corner_width = height // 4, width // 4
    corners = [
        image[:corner_height, :corner_width],
        image[:corner_height, -corner_width:],
        image[-corner_height:, :corner_width],
        image[-corner_height:, -corner_width:],
    ]
    estimates = []
    for corner in corners:
        pixels = corner[np.isfinite(corner)].astype(np.float64)
        if pixels.size < 2:
            continue
        rms = float(np.sqrt(np.mean(pixels**2)))
        if np.isfinite(rms) and rms > 0:
            estimates.append(rms)

    return min(estimates) if estimates else float("nan")


def get_noise_jy_per_beam(image_hdu: fits.PrimaryHDU, method: str) -> float:
    """Select a noise estimate in Jy/beam; reject missing or unusable estimates."""
    if method == "header":
        unit = str(image_hdu.header.get("BUNIT", "")).strip()
        if unit.replace(" ", "").upper() != "JY/BEAM":
            raise ValueError(f"Expected image units of Jy/beam; got {unit!r}")
        try:
            noise = float(image_hdu.header.get("NOISE", np.nan))
        except (TypeError, ValueError) as error:
            raise ValueError("Header NOISE must be finite and positive, in Jy/beam.") from error
    elif method == "four_corner":
        noise = four_corner_rms_jy_per_beam(image_hdu)
    else:
        raise ValueError(f"Unknown noise method {method!r}; expected 'header' or 'four_corner'.")

    if not np.isfinite(noise) or noise <= 0:
        raise ValueError(f"Noise method {method!r} did not provide a finite positive RMS in Jy/beam.")
    return float(noise)


# Lister et al. (2013), Section 5.2: flux-weighted CLEAN-component PA,
# within some mas region of the core and above some sigma times the image rms.
# https://doi.org/10.1088/0004-6256/146/5/120
# Adaptation: the core is fixed at the geometric image centre.
# Numerical choice: a circular mean handles angle wraparound.
# The published cut compares component flux in Jy with image rms in Jy/beam
# numerically; it is not a conversion between these two units.
def jet_orientation_deg(
    image_hdu: fits.PrimaryHDU,
    clean_components_hdu: fits.BinTableHDU,
    noise_jy_per_beam: float,
    inner_mas: float = 0.15,
    outer_mas: float = 1.0,
    sigma_cut: float = 3.0,
) -> float:
    """Return inner-jet PA in [0, 360), north through east; NaN if undefined."""
    header = image_hdu.header
    unit = str(header.get("BUNIT", "")).strip()
    if unit.replace(" ", "").upper() != "JY/BEAM":
        raise ValueError(f"Expected image units of Jy/beam; got {unit!r}")
    if not np.isfinite(noise_jy_per_beam) or noise_jy_per_beam <= 0:
        raise ValueError("Image noise must be finite and positive, in Jy/beam.")
    if (
        not np.isfinite([inner_mas, outer_mas, sigma_cut]).all()
        or not 0 <= inner_mas < outer_mas
        or sigma_cut <= 0
    ):
        raise ValueError("Require finite 0 <= inner_mas < outer_mas and sigma_cut > 0.")

    if str(clean_components_hdu.header.get("EXTNAME", "")).strip().upper() != "AIPS CC":
        raise ValueError("Expected an AIPS CC table.")
    for column, allowed_units in [
        ("FLUX", {"JY"}),
        ("DELTAX", {"DEG", "DEGREE", "DEGREES"}),
        ("DELTAY", {"DEG", "DEGREE", "DEGREES"}),
    ]:
        if column not in clean_components_hdu.columns.names:
            raise ValueError(f"Missing {column} column in CLEAN-component table.")
        column_unit = (clean_components_hdu.columns[column].unit or "").strip().upper()
        if column_unit not in allowed_units:
            raise ValueError(f"Unexpected units for CLEAN-component {column}: {column_unit!r}")
    if clean_components_hdu.data is None or len(clean_components_hdu.data) == 0:
        return float("nan")

    flux = np.asarray(clean_components_hdu.data["FLUX"], dtype=np.float64)
    dx = np.asarray(clean_components_hdu.data["DELTAX"], dtype=np.float64)
    dy = np.asarray(clean_components_hdu.data["DELTAY"], dtype=np.float64)
    if not (np.isfinite(flux).all() and np.isfinite(dx).all() and np.isfinite(dy).all()):
        raise ValueError("Non-finite CLEAN-component data.")

    # AIPS CC offsets are in degrees along the image axes, relative to CRPIX.
    x_pixel = dx / header["CDELT1"] + header["CRPIX1"] - 1
    y_pixel = dy / header["CDELT2"] + header["CRPIX2"] - 1
    wcs = WCS(header).celestial
    core_x_pixel = (header["NAXIS1"] - 1) / 2
    core_y_pixel = (header["NAXIS2"] - 1) / 2
    core = wcs.pixel_to_world(core_x_pixel, core_y_pixel)
    positions = wcs.pixel_to_world(x_pixel, y_pixel)
    radius_mas = core.separation(positions).mas
    keep = (
        (radius_mas >= inner_mas) & (radius_mas <= outer_mas)
        & (flux > sigma_cut * noise_jy_per_beam)
    )
    if not keep.any():
        return float("nan")

    angles = core.position_angle(positions[keep]).rad
    weights = flux[keep] / flux[keep].sum()
    east = np.sum(weights * np.sin(angles))
    north = np.sum(weights * np.cos(angles))
    if np.hypot(east, north) <= 1e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(east, north)) % 360)


# Pushkarev et al. (2017), Section 3.1: split the above-8-sigma arc
# intensity equally between two sides of the ridge position.
# https://arxiv.org/pdf/1705.02888
# Numerical choices: bilinear interpolation, uniform angular samples,
# and a discrete intensity-weighted median. The angular window defaults
# to +/-90 degrees around the supplied direction; this is configurable.
def sample_jet_arc(
    image_hdu: fits.PrimaryHDU,
    radius_mas: float,
    noise_jy_per_beam: float,
    pa_center_deg: float,
    half_width_deg: float = 90.0,
    angle_step_deg: float = 0.5,
    sigma_cut: float = 8.0,
) -> dict[str, float | bool] | None:
    """Return one ridge point, or None without qualifying emission."""
    settings = [
        radius_mas, noise_jy_per_beam, pa_center_deg,
        half_width_deg, angle_step_deg, sigma_cut,
    ]
    if not np.isfinite(settings).all():
        raise ValueError("Arc settings and noise must be finite.")
    if radius_mas <= 0 or noise_jy_per_beam <= 0 or angle_step_deg <= 0 or sigma_cut <= 0:
        raise ValueError("Radius, noise, angular step and sigma cut must be positive.")
    if not 0 < half_width_deg < 180:
        raise ValueError("Arc half-width must be between 0 and 180 degrees.")
    header = image_hdu.header
    if str(header.get("BUNIT", "")).replace(" ", "").upper() != "JY/BEAM":
        raise ValueError("Arc sampling requires image units of Jy/beam.")
    if image_hdu.data is None:
        raise ValueError("Image HDU contains no pixel data.")
    image = np.squeeze(image_hdu.data)
    if image.ndim != 2:
        raise ValueError("Arc sampling requires a 2D image.")

    sample_count = int(np.ceil(2 * half_width_deg / angle_step_deg)) + 1
    angles = np.linspace(
        pa_center_deg - half_width_deg, pa_center_deg + half_width_deg, sample_count,
    )
    wcs = WCS(header).celestial
    height, width = image.shape
    core = wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)
    positions = core.directional_offset_by(angles * u.deg, radius_mas * u.mas)
    x_pixel, y_pixel = wcs.world_to_pixel(positions)
    inside = (
        np.isfinite(x_pixel) & np.isfinite(y_pixel)
        & (x_pixel >= 0) & (x_pixel <= width - 1)
        & (y_pixel >= 0) & (y_pixel <= height - 1)
    )
    brightness = map_coordinates(
        image, [y_pixel, x_pixel], order=1, output=np.float64,
        mode="constant", cval=np.nan, prefilter=False,
    )
    detected = (
        inside & np.isfinite(brightness)
        & (brightness >= sigma_cut * noise_jy_per_beam)
    )
    if not detected.any():
        return None

    # Equal angular spacing at fixed radius gives equal arc-length weights.
    weights = np.where(detected, brightness, 0.0)
    index = int(np.searchsorted(np.cumsum(weights), weights.sum() / 2))
    return {
        "radius_mas": float(radius_mas),
        "pa_deg": float(angles[index] % 360),
        "x_pixel": float(x_pixel[index]),
        "y_pixel": float(y_pixel[index]),
        "arc_clipped": bool(not inside.all()),
    }


# Increasing-radius arc tracing follows Pushkarev et al. (2017), Section 3.1.
# https://arxiv.org/pdf/1705.02888
# Implementation choices: a moving angular window centred on the previous
# point and stopping at the first undetected arc. The raw radial step is
# configurable; the paper's approximately 0.05 mas refers to its spline.
def trace_jet_ridgeline(
    image_hdu: fits.PrimaryHDU,
    noise_jy_per_beam: float,
    initial_pa_deg: float,
    radial_step_mas: float = 0.05,
    max_radius_mas: float | None = None,
    half_width_deg: float = 90.0,
    angle_step_deg: float = 0.5,
    sigma_cut: float = 8.0,
) -> tuple[list[dict[str, float | bool]], str]:
    """Return temporary ridge points and a stopping reason on the full image."""
    if image_hdu.data is None:
        raise ValueError("Image HDU contains no pixel data.")
    image = np.squeeze(image_hdu.data)
    if image.ndim != 2:
        raise ValueError("Ridgeline tracing requires a 2D image.")
    if not np.isfinite(initial_pa_deg):
        raise ValueError("A finite initial jet position angle is required.")
    if not np.isfinite(radial_step_mas) or radial_step_mas <= 0:
        raise ValueError("Radial step must be finite and positive, in mas.")

    stopping_reason = "max_radius"
    if max_radius_mas is None:
        height, width = image.shape
        wcs = WCS(image_hdu.header).celestial
        core = wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)
        corners = wcs.pixel_to_world(
            [0, width - 1, 0, width - 1], [0, 0, height - 1, height - 1],
        )
        max_radius_mas = float(core.separation(corners).mas.max())
        stopping_reason = "image_radius_limit"
    if not np.isfinite(max_radius_mas) or max_radius_mas < radial_step_mas:
        raise ValueError("Maximum radius must be finite and at least one radial step.")

    points = []
    pa_center = float(initial_pa_deg)
    step_count = int(np.floor(max_radius_mas / radial_step_mas))
    for step in range(1, step_count + 1):
        point = sample_jet_arc(
            image_hdu, step * radial_step_mas, noise_jy_per_beam, pa_center,
            half_width_deg=half_width_deg, angle_step_deg=angle_step_deg,
            sigma_cut=sigma_cut,
        )
        if point is None:
            stopping_reason = "no_detection"
            break
        points.append(point)
        pa_center = point["pa_deg"]

    return points, stopping_reason


# Pushkarev et al. (2017), Section 3.1: cubic-spline ridge interpolation
# at approximately 0.05 mas radial intervals.
# https://arxiv.org/pdf/1705.02888
# Numerical choices: interpolate pixel x and y against sampled arc radius,
# with natural boundary conditions and no extrapolation.
# The radius parameter labels the interpolation grid; intermediate spline
# positions are not new detections or independent radial measurements.
def interpolate_ridgeline(
    ridge: list[dict[str, float | bool]],
    radial_step_mas: float = 0.05,
) -> list[dict[str, float]]:
    """Return inspection points within the detected range, preserving endpoints."""
    if not np.isfinite(radial_step_mas) or radial_step_mas <= 0:
        raise ValueError("Interpolation step must be finite and positive, in mas.")
    values = np.asarray([
        [point["radius_mas"], point["x_pixel"], point["y_pixel"]]
        for point in ridge
    ], dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(values).all():
        raise ValueError("Ridgeline coordinates and radii must be finite.")
    radius = values[:, 0]
    if (np.diff(radius) <= 0).any():
        raise ValueError("Ridgeline radii must be strictly increasing.")

    if len(values) < 2:
        sample_radius = radius
        coordinates = values[:, 1:]
    else:
        step_count = int(np.floor((radius[-1] - radius[0]) / radial_step_mas))
        sample_radius = radius[0] + radial_step_mas * np.arange(step_count + 1)
        sample_radius = np.append(sample_radius[sample_radius < radius[-1]], radius[-1])
        spline = CubicSpline(
            radius, values[:, 1:], axis=0, bc_type="natural", extrapolate=False,
        )
        coordinates = spline(sample_radius)

    return [
        {
            "radius_parameter_mas": float(sample),
            "x_pixel": float(x_pixel),
            "y_pixel": float(y_pixel),
        }
        for sample, (x_pixel, y_pixel) in zip(sample_radius, coordinates)
    ]


def iter_observations(
    input_dir: str,
    uv_dir: str,
) -> Iterator[tuple[Path, Path | None]]:
    """Yield image paths paired with their UV path, or None if unavailable."""
    for image_path in sorted(Path(input_dir).glob("*.icn.fits.gz")):
        if not image_path.is_file():
            continue
        observation_name = image_path.name.removesuffix(".icn.fits.gz")
        uv_path = Path(uv_dir) / f"{observation_name}.uvf"
        yield image_path, uv_path if uv_path.is_file() else None


def process_observation(
    observation: tuple[Path, Path | None],
    img_size: int,
    output_dir: Path,
) -> None:
    """Open one FITS/UV pair, calculate its metadata and save the crop."""
    image_path, uv_path = observation
    with (
        fits.open(image_path) as image_hdul,
        (fits.open(uv_path) if uv_path is not None else nullcontext()) as uv_hdul,
    ):
        noise = get_noise_jy_per_beam(image_hdul[0], NOISE_METHOD)
        jet_pa = jet_orientation_deg(
            image_hdul[0], image_hdul["AIPS CC"], noise,
            **JET_ORIENTATION_SETTINGS,
        )
        ridge, stopping_reason = trace_jet_ridgeline(
            image_hdul[0], noise, initial_pa_deg=jet_pa,
            **RIDGELINE_SETTINGS,
        )
        image = torch.from_numpy(
            image_hdul[0].data[0, 0].astype("float32")
        )
        cropped_image = center_crop(image, img_size).clone()
        metadata = {
            "source_name": image_path.name.split(".", 1)[0],
            "image_size": list(cropped_image.shape),
            "noise_rms_jy_per_beam": noise,
            "noise_method": NOISE_METHOD,
            "peak_brightness_jy_per_beam": peak_brightness_jy_per_beam(image_hdul[0]),
            "jet_pa_deg": jet_pa,
            "jet_radial_extent_mas": ridge[-1]["radius_mas"] if ridge else float("nan"),
            "ridgeline_stop_reason": stopping_reason,
            # Whether any detected ridge point's search arc crossed the image boundary.
            "ridgeline_arc_clipped": any(point["arc_clipped"] for point in ridge),
        }
        # Add feature calculations here using image_hdul and uv_hdul.
        # uv_hdul is None when the observation has no matching UV file.

    output_name = image_path.name.replace(".icn.fits.gz", ".icn.pt")
    sample = {
        "image": cropped_image,
        "metadata": metadata,
    }
    torch.save(sample, output_dir / output_name)


@timelogger
def preprocess_data(
    input_dir: str | None = None,
    uv_dir: str | None = None,
    img_size: int = 256,
    cpu_cores: int | None = None,
    output_dir: str | None = None,
) -> None:
    if cpu_cores is None:
        cpu_cores = get_preprocess_cpu_cores()
    if img_size < 1 or cpu_cores < 1:
        raise ValueError("Image size and CPU worker count must be positive.")

    input_dir = get_path("MOJAVE_FITS_DIR") if input_dir is None else Path(input_dir).expanduser()
    uv_dir = get_path("MOJAVE_UV_DIR") if uv_dir is None else Path(uv_dir).expanduser()
    output_dir = (
        get_mojave_data_dir(img_size)
        if output_dir is None else Path(output_dir).expanduser()
    )
    print(
        f"MOJAVE preprocessing\nFITS directory: {input_dir}\nUV directory: {uv_dir}\n"
        f"Output directory: {output_dir}\nImage size: {img_size}; CPU cores: {cpu_cores}",
        flush=True,
    )
    observations = list(iter_observations(input_dir, uv_dir))
    if not observations:
        raise FileNotFoundError(f"No FITS files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    process_image = partial(
        process_observation,
        img_size=img_size,
        output_dir=output_dir,
    )

    missing_uv = [image.name for image, uv in observations if uv is None]
    processed_count = 0
    failures = []
    with ProcessPoolExecutor(
        max_workers=cpu_cores,
        initializer=torch.set_num_threads,
        initargs=(1,),
    ) as executor, tqdm(
        total=len(observations), desc="Processing observations", unit="image",
    ) as progress:
        futures = {
            executor.submit(process_image, observation): observation[0]
            for observation in observations
        }
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((image_path.name, f"{type(error).__name__}: {error}"))
            else:
                processed_count += 1
            progress.update(1)

    print(f"\nImages found: {len(observations)}")
    print(f"Images processed and saved: {processed_count}")
    print(f"Images without matching UV files: {len(missing_uv)}")
    print(f"Processing failures: {len(failures)}")
    if missing_uv:
        print("Images without matching UV files:")
        for filename in missing_uv:
            print(f"  {filename}")
    if failures:
        print("Failed images:")
        for filename, error in sorted(failures):
            print(f"  {filename}: {error}")
        raise RuntimeError(f"Preprocessing failed for {len(failures)} images; see report above.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", default=None, help="FITS directory; defaults to MOJAVE_FITS_DIR in .env.",
    )
    parser.add_argument(
        "--uv_dir", default=None, help="UV directory; defaults to MOJAVE_UV_DIR in .env.",
    )
    parser.add_argument("--output_dir", default=None, help="Defaults to MOJAVE_DATA_ROOT/Mojave_I_<image_size>.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--cpu_cores", type=int, default=None, help="Defaults to MOJAVE_PREPROCESS_CPUS in .env.")
    args = parser.parse_args()

    if args.image_size < 1 or (args.cpu_cores is not None and args.cpu_cores < 1):
        parser.error("--image_size and --cpu_cores must be positive.")
    
    print("Preprocessing is ignoring the following warning if it appears: \n fitsfixedwarning datfix made the change")

    preprocess_data(
        args.input_dir,
        args.uv_dir,
        img_size=args.image_size,
        cpu_cores=args.cpu_cores,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
