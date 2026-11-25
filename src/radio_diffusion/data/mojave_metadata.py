"""Extract MOJAVE metadata without scaling or transforming image intensities.

Image metrics prefixed with ``cropped_`` describe the saved central crop.
Image metrics without that prefix describe the full image before cropping.
The structural ``image_size`` field records the saved crop dimensions.
"""

from dataclasses import dataclass

import numpy as np
import torch
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class JetOrientationSettings:
    """Full-image CLEAN-component selection for the inner-jet direction."""

    inner_mas: float
    outer_mas: float
    sigma_cut: float


@dataclass(frozen=True)
class RidgelineSettings:
    """Radial tracing and angular sampling of the full-image jet ridgeline."""

    radial_step_mas: float
    max_radius_mas: float | None
    half_width_deg: float
    angle_step_deg: float
    sigma_cut: float


METADATA_SCHEMA = {
    "source_name": {"type": "string"},
    "image_size": {"type": "integer_pair", "unit": "pixels"},
    "spatial_geometry": {"type": "object"},
    "noise_rms_jy_per_beam": {"type": "number", "unit": "Jy/beam"},
    "noise_method": {"type": "string"},
    "peak_brightness_jy_per_beam": {"type": "number", "unit": "Jy/beam"},
    "jet_pa_deg": {"type": "number", "unit": "degrees", "may_be_nonfinite": True},
    "jet_radial_extent_mas": {"type": "number", "unit": "mas", "may_be_nonfinite": True},
    "ridgeline_stop_reason": {"type": "string"},
    "ridgeline_arc_clipped": {"type": "boolean"},
    "cropped_image_pixel_maximum": {"type": "number", "unit": "Jy/beam"},
    "cropped_image_pixel_minimum": {"type": "number", "unit": "Jy/beam"},
    "cropped_image_pixel_mean": {"type": "number", "unit": "Jy/beam"},
    "cropped_image_pixel_std": {"type": "number", "unit": "Jy/beam"},
}


def _brightness_image(image_hdu: fits.PrimaryHDU):
    """Require image data expressed in Jy/beam, retaining its original shape."""
    unit = str(image_hdu.header.get("BUNIT", "")).strip()
    if unit.replace(" ", "").upper() != "JY/BEAM":
        raise ValueError(f"Expected image units of Jy/beam; got {unit!r}")
    if image_hdu.data is None:
        raise ValueError("Image HDU contains no pixel data.")
    return image_hdu.data


# Kovalev et al. (2005), Section 4: signed sum of all CLEAN-model fluxes.
# https://arxiv.org/pdf/astro-ph/0505536
# AIPS Memo 117, Section 6.1: component FLUX is in Jy.
# https://www.aips.nrao.edu/TEXT/PUBL/AIPSMEM117.PDF
def total_flux_jy(clean_components_hdu: fits.BinTableHDU) -> float:
    """Return signed full CLEAN-model flux before cropping, without a threshold."""
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


# Jy/beam brightness convention: Kovalev et al. (2005), Fig. 1.
# https://arxiv.org/pdf/astro-ph/0505536
def peak_brightness_jy_per_beam(image_hdu: fits.PrimaryHDU) -> float:
    """Return maximum non-blank full-image brightness before cropping."""
    image = _brightness_image(image_hdu)
    if np.isinf(image).any() or not np.isfinite(image).any():
        raise ValueError("Image contains infinite values or has no finite pixels.")
    return float(np.nanmax(image))


# Pushkarev et al. (2017), Section 2.1: minimum RMS of four corner regions,
# each covering 1/16 of the image area. RMS about zero; no clipping.
# https://arxiv.org/pdf/1705.02888
def four_corner_rms_jy_per_beam(image_hdu: fits.PrimaryHDU) -> float:
    """Return minimum positive full-image corner RMS, or NaN if unavailable."""
    image = np.squeeze(_brightness_image(image_hdu))
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
    """Select a noise estimate and require a finite positive RMS in Jy/beam."""
    if method == "header":
        _brightness_image(image_hdu)
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


def _validate_orientation_settings(noise, settings: JetOrientationSettings) -> None:
    """Require finite orientation settings and a positive noise estimate."""
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Image noise must be finite and positive, in Jy/beam.")
    if (
        not np.isfinite([settings.inner_mas, settings.outer_mas, settings.sigma_cut]).all()
        or not 0 <= settings.inner_mas < settings.outer_mas
        or settings.sigma_cut <= 0
    ):
        raise ValueError("Require finite 0 <= inner_mas < outer_mas and sigma_cut > 0.")


def _clean_component_offsets(clean_components_hdu):
    """Validate the CLEAN table and return flux, dx and dy arrays, or None."""
    if str(clean_components_hdu.header.get("EXTNAME", "")).strip().upper() != "AIPS CC":
        raise ValueError("Expected an AIPS CC table.")
    for column, allowed_units in [
        ("FLUX", {"JY"}),
        ("DELTAX", {"DEG", "DEGREE", "DEGREES"}),
        ("DELTAY", {"DEG", "DEGREE", "DEGREES"}),
    ]:
        if column not in clean_components_hdu.columns.names:
            raise ValueError(f"Missing {column} column in CLEAN-component table.")
        unit = (clean_components_hdu.columns[column].unit or "").strip().upper()
        if unit not in allowed_units:
            raise ValueError(f"Unexpected units for CLEAN-component {column}: {unit!r}")
    if clean_components_hdu.data is None or len(clean_components_hdu.data) == 0:
        return None
    arrays = tuple(
        np.asarray(clean_components_hdu.data[column], dtype=np.float64)
        for column in ("FLUX", "DELTAX", "DELTAY")
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Non-finite CLEAN-component data.")
    return arrays


def _component_polar_coordinates(image_hdu, dx, dy):
    """Locate CLEAN components relative to the geometric full-image centre."""
    header = image_hdu.header
    # AIPS CC offsets are in degrees along the axes, relative to CRPIX.
    x_pixel = dx / header["CDELT1"] + header["CRPIX1"] - 1
    y_pixel = dy / header["CDELT2"] + header["CRPIX2"] - 1
    wcs = WCS(header).celestial
    core = wcs.pixel_to_world((header["NAXIS1"] - 1) / 2, (header["NAXIS2"] - 1) / 2)
    positions = wcs.pixel_to_world(x_pixel, y_pixel)
    return core.separation(positions).mas, core.position_angle(positions).rad


# Lister et al. (2013), Section 5.2: flux-weighted CLEAN-component PA.
# https://doi.org/10.1088/0004-6256/146/5/120
# The core is fixed at the geometric image centre; angles use a circular mean.
# The flux-in-Jy versus rms-in-Jy/beam cut is numerical, not a unit conversion.
def jet_orientation_deg(
    image_hdu: fits.PrimaryHDU,
    clean_components_hdu: fits.BinTableHDU,
    noise_jy_per_beam: float,
    settings: JetOrientationSettings,
) -> float:
    """Return inner-jet PA in [0, 360), north through east; NaN if undefined."""
    _brightness_image(image_hdu)
    _validate_orientation_settings(noise_jy_per_beam, settings)
    components = _clean_component_offsets(clean_components_hdu)
    if components is None:
        return float("nan")
    flux, dx, dy = components
    radius_mas, angles = _component_polar_coordinates(image_hdu, dx, dy)
    keep = (
        (radius_mas >= settings.inner_mas) & (radius_mas <= settings.outer_mas)
        & (flux > settings.sigma_cut * noise_jy_per_beam)
    )
    if not keep.any():
        return float("nan")
    weights = flux[keep] / flux[keep].sum()
    east = np.sum(weights * np.sin(angles[keep]))
    north = np.sum(weights * np.cos(angles[keep]))
    if np.hypot(east, north) <= 1e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(east, north)) % 360)


def _validate_arc_settings(radius, noise, pa, settings: RidgelineSettings) -> None:
    """Check the finite angular and brightness settings used to sample an arc."""
    values = [radius, noise, pa, settings.half_width_deg, settings.angle_step_deg, settings.sigma_cut]
    if not np.isfinite(values).all():
        raise ValueError("Arc settings and noise must be finite.")
    if radius <= 0 or noise <= 0 or settings.angle_step_deg <= 0 or settings.sigma_cut <= 0:
        raise ValueError("Radius, noise, angular step and sigma cut must be positive.")
    if not 0 < settings.half_width_deg < 180:
        raise ValueError("Arc half-width must be between 0 and 180 degrees.")


def _sample_arc_brightness(image_hdu, radius_mas, pa_center_deg, settings: RidgelineSettings):
    """Sample full-image brightness along a celestial arc with bilinear interpolation."""
    image = np.squeeze(_brightness_image(image_hdu))
    if image.ndim != 2:
        raise ValueError("Arc sampling requires a 2D image.")
    sample_count = int(np.ceil(2 * settings.half_width_deg / settings.angle_step_deg)) + 1
    angles = np.linspace(
        pa_center_deg - settings.half_width_deg, pa_center_deg + settings.half_width_deg, sample_count,
    )
    wcs = WCS(image_hdu.header).celestial
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
    return angles, x_pixel, y_pixel, inside, brightness


# Pushkarev et al. (2017), Section 3.1: split above-threshold arc intensity
# equally between the sides of the ridge position, using a weighted median.
# https://arxiv.org/pdf/1705.02888
def sample_jet_arc(
    image_hdu: fits.PrimaryHDU,
    radius_mas: float,
    noise_jy_per_beam: float,
    pa_center_deg: float,
    settings: RidgelineSettings,
) -> dict[str, float | bool] | None:
    """Return one ridge point, or None without qualifying emission."""
    _validate_arc_settings(radius_mas, noise_jy_per_beam, pa_center_deg, settings)
    angles, x_pixel, y_pixel, inside, brightness = _sample_arc_brightness(
        image_hdu, radius_mas, pa_center_deg, settings,
    )
    detected = inside & np.isfinite(brightness) & (brightness >= settings.sigma_cut * noise_jy_per_beam)
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


def _ridgeline_radius_limit(image_hdu) -> float:
    """Find the greatest full-image corner distance from the geometric centre."""
    height, width = np.squeeze(image_hdu.data).shape
    wcs = WCS(image_hdu.header).celestial
    core = wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)
    corners = wcs.pixel_to_world(
        [0, width - 1, 0, width - 1], [0, 0, height - 1, height - 1],
    )
    return float(core.separation(corners).mas.max())


def _validate_ridgeline_inputs(image_hdu, initial_pa_deg, radial_step_mas) -> None:
    """Require a 2D full image, finite initial direction and positive step."""
    if image_hdu.data is None:
        raise ValueError("Image HDU contains no pixel data.")
    if np.squeeze(image_hdu.data).ndim != 2:
        raise ValueError("Ridgeline tracing requires a 2D image.")
    if not np.isfinite(initial_pa_deg):
        raise ValueError("A finite initial jet position angle is required.")
    if not np.isfinite(radial_step_mas) or radial_step_mas <= 0:
        raise ValueError("Radial step must be finite and positive, in mas.")


# Increasing-radius arc tracing: Pushkarev et al. (2017), Section 3.1.
# A moving window follows the previous point and stops at the first gap.
# https://arxiv.org/pdf/1705.02888
def trace_jet_ridgeline(
    image_hdu: fits.PrimaryHDU,
    noise_jy_per_beam: float,
    initial_pa_deg: float,
    settings: RidgelineSettings,
) -> tuple[list[dict[str, float | bool]], str]:
    """Return temporary ridge points and a stopping reason on the full image."""
    _validate_ridgeline_inputs(image_hdu, initial_pa_deg, settings.radial_step_mas)
    max_radius_mas = settings.max_radius_mas
    stopping_reason = "max_radius"
    if max_radius_mas is None:
        max_radius_mas = _ridgeline_radius_limit(image_hdu)
        stopping_reason = "image_radius_limit"
    if not np.isfinite(max_radius_mas) or max_radius_mas < settings.radial_step_mas:
        raise ValueError("Maximum radius must be finite and at least one radial step.")
    points = []
    pa_center = float(initial_pa_deg)
    step_count = int(np.floor(max_radius_mas / settings.radial_step_mas))
    for step in range(1, step_count + 1):
        point = sample_jet_arc(
            image_hdu, step * settings.radial_step_mas, noise_jy_per_beam, pa_center, settings,
        )
        if point is None:
            stopping_reason = "no_detection"
            break
        points.append(point)
        pa_center = point["pa_deg"]
    return points, stopping_reason


def _interpolate_ridge_coordinates(values, radial_step_mas):
    """Interpolate pixel coordinates within the detected radius range."""
    radius = values[:, 0]
    if len(values) < 2:
        return radius, values[:, 1:]
    step_count = int(np.floor((radius[-1] - radius[0]) / radial_step_mas))
    sample_radius = radius[0] + radial_step_mas * np.arange(step_count + 1)
    sample_radius = np.append(sample_radius[sample_radius < radius[-1]], radius[-1])
    spline = CubicSpline(radius, values[:, 1:], axis=0, bc_type="natural", extrapolate=False)
    return sample_radius, spline(sample_radius)


# Pushkarev et al. (2017), Section 3.1: cubic-spline inspection points.
# Pixel x/y are interpolated against sampled arc radius, with natural boundary
# conditions and no extrapolation. These points are not independent detections.
# https://arxiv.org/pdf/1705.02888
def interpolate_ridgeline(
    ridge: list[dict[str, float | bool]], settings: RidgelineSettings,
) -> list[dict[str, float]]:
    """Return inspection points within the detected range, preserving endpoints."""
    radial_step_mas = settings.radial_step_mas
    if not np.isfinite(radial_step_mas) or radial_step_mas <= 0:
        raise ValueError("Interpolation step must be finite and positive, in mas.")
    values = np.asarray([
        [point["radius_mas"], point["x_pixel"], point["y_pixel"]] for point in ridge
    ], dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(values).all():
        raise ValueError("Ridgeline coordinates and radii must be finite.")
    if (np.diff(values[:, 0]) <= 0).any():
        raise ValueError("Ridgeline radii must be strictly increasing.")
    sample_radius, coordinates = _interpolate_ridge_coordinates(values, radial_step_mas)
    return [
        {
            "radius_parameter_mas": float(sample),
            "x_pixel": float(x_pixel),
            "y_pixel": float(y_pixel),
        }
        for sample, (x_pixel, y_pixel) in zip(sample_radius, coordinates)
    ]


def full_image_metadata(
    image_hdul, source_name: str, noise_method: str,
    orientation_settings: JetOrientationSettings, ridgeline_settings: RidgelineSettings,
) -> dict:
    """Calculate full-image metrics with the explicit preprocessing settings."""
    noise = get_noise_jy_per_beam(image_hdul[0], noise_method)
    jet_pa = jet_orientation_deg(
        image_hdul[0], image_hdul["AIPS CC"], noise, orientation_settings,
    )
    ridge, stopping_reason = trace_jet_ridgeline(
        image_hdul[0], noise, initial_pa_deg=jet_pa, settings=ridgeline_settings,
    )
    return {
        "source_name": source_name,
        "noise_rms_jy_per_beam": noise,
        "noise_method": noise_method,
        "peak_brightness_jy_per_beam": peak_brightness_jy_per_beam(image_hdul[0]),
        "jet_pa_deg": jet_pa,
        "jet_radial_extent_mas": ridge[-1]["radius_mas"] if ridge else float("nan"),
        "ridgeline_stop_reason": stopping_reason,
        "ridgeline_arc_clipped": any(point["arc_clipped"] for point in ridge),
    }


def crop_spatial_geometry(header: fits.Header, left: int, top: int) -> dict:
    """Map zero-based crop (x, y) offsets to local east/north offsets in mas."""
    x = (header["NAXIS1"] - 1) / 2
    y = (header["NAXIS2"] - 1) / 2
    wcs = WCS(header).celestial
    centre = wcs.pixel_to_world(x, y)
    neighbours = wcs.pixel_to_world([x + 1, x], [y, y + 1])
    east, north = centre.spherical_offsets_to(neighbours)
    matrix = np.stack([east.mas, north.mas])
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
        raise ValueError("Crop geometry requires a finite, invertible celestial pixel scale.")
    return {
        "reference_pixel_xy": [x - left, y - top],
        "pixel_to_east_north_mas": matrix.tolist(),
    }


def cropped_image_metadata(cropped_image: torch.Tensor) -> dict:
    """Describe the exact saved crop with a population standard deviation."""
    pixels = cropped_image.to(torch.float64)
    return {
        "image_size": list(cropped_image.shape),
        "cropped_image_pixel_maximum": pixels.max().item(),
        "cropped_image_pixel_minimum": pixels.min().item(),
        "cropped_image_pixel_mean": pixels.mean().item(),
        "cropped_image_pixel_std": pixels.std(correction=0).item(),
    }
