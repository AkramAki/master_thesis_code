"""Shared MOJAVE image plots for notebooks and saved sample grids."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, PowerNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch
from matplotlib.transforms import Affine2D
from scipy.interpolate import PchipInterpolator


ALLOWED_COLORMAPS = ("inferno", "magma", "viridis", "gray")


@dataclass(frozen=True)
class ImageStyle:
    cmap: str = "inferno"
    scale: Literal["linear", "log", "gamma"] = "gamma"
    gamma: float = 0.5

    def __post_init__(self):
        if self.cmap not in ALLOWED_COLORMAPS:
            raise ValueError(f"cmap must be one of {', '.join(ALLOWED_COLORMAPS)}.")
        if self.scale not in ("linear", "log", "gamma"):
            raise ValueError("scale must be 'linear', 'log' or 'gamma'.")
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive.")


IMAGE_STYLES = {
    "default": ImageStyle(),
    "linear": ImageStyle(scale="linear"),
    "log": ImageStyle(scale="log"),
}


@dataclass(frozen=True)
class ImageCoordinates:
    """Map (column, row) offsets from a reference pixel into the named unit."""

    unit: str
    reference_pixel_xy: tuple[float, float] | None = None
    pixel_to_units: tuple[tuple[float, float], tuple[float, float]] | None = None
    x_label: str = "X"
    y_label: str = "Y"
    x_increases_left: bool = False
    y_increases_down: bool | None = None

    def __post_init__(self):
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("Image coordinates require a unit name.")
        reference, matrix = self.reference_pixel_xy, self.pixel_to_units
        if reference is None and matrix is None and self.unit == "pixels":
            reference, matrix = (0, 0), ((1, 0), (0, 1))
        if reference is not None or matrix is not None:
            reference = np.asarray(reference, dtype=float)
            matrix = np.asarray(matrix, dtype=float)
            if (
                reference.shape != (2,) or matrix.shape != (2, 2)
                or not np.isfinite(reference).all() or not np.isfinite(matrix).all()
                or np.linalg.det(matrix) == 0
            ):
                raise ValueError("Image coordinates need a finite reference and invertible 2x2 matrix.")
            object.__setattr__(self, "reference_pixel_xy", tuple(reference.tolist()))
            object.__setattr__(self, "pixel_to_units", tuple(map(tuple, matrix.tolist())))
        if self.y_increases_down is None:
            object.__setattr__(self, "y_increases_down", self.unit == "pixels")

    def _calibration(self):
        if self.pixel_to_units is None:
            raise ValueError(f"Mapping pixels to {self.unit!r} requires a reference and a matrix.")
        return np.asarray(self.reference_pixel_xy), np.asarray(self.pixel_to_units)

    def from_pixels(self, pixel_xy):
        """Convert one or several (column, row) positions into axes coordinates."""
        reference, matrix = self._calibration()
        return (np.asarray(pixel_xy) - reference) @ matrix.T


_PIXEL_COORDINATES = ImageCoordinates(unit="pixels")


def _overlay_points(points, *, min_points=2):
    points = np.array(points, dtype=float, copy=True)
    if (
        points.ndim != 2 or points.shape[1] != 2 or len(points) < min_points
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"An overlay requires at least {min_points} finite (x, y) points.")
    return points


@dataclass(frozen=True)
class _OverlayItem:
    description: str
    draw: Callable[..., Artist]
    label: str | None = None
    required_unit: str | None = None


def _point_text(point):
    return f"({point[0]:g}, {point[1]:g})"


def _faded_arc_artist(segments, *, color, alpha, **style):
    colors = np.tile(to_rgba(color, alpha), (len(segments), 1))
    colors[:, 3] *= (1 + np.cos(np.linspace(-np.pi, np.pi, len(segments)))) / 2
    style.setdefault("zorder", 1)
    return LineCollection(segments, colors=colors, **style)


@dataclass(frozen=True)
class ImageOverlay:
    """Collect shapes in one coordinate system and draw them in order.

    Shapes accept label, color and alpha. Lines and outlines also accept
    linewidth and linestyle.
    """

    coordinates: ImageCoordinates
    _items: list[_OverlayItem] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.coordinates, ImageCoordinates):
            raise TypeError("ImageOverlay requires ImageCoordinates.")

    def append(self, item):
        """Add a configured shape to this image's overlays."""
        if not isinstance(item, _OverlayItem):
            raise TypeError("Append a shape created by an overlay constructor.")
        if item.required_unit is not None and item.required_unit != self.coordinates.unit:
            raise ValueError(f"{item.description} requires {item.required_unit} axes.")
        self._items.append(item)

    def summarize(self):
        """Print the configured shapes in drawing order."""
        print(f"{len(self._items)} overlays ({self.coordinates.unit})")
        for index, item in enumerate(self._items, start=1):
            label = "" if not item.label else f" [{item.label}]"
            print(f"{index}. {item.description}{label}")

    def overlay(self, image=None, *, ax=None, **plot_options):
        """Return the axes after drawing an image or adding to an existing one.

        For axes created outside this module, the caller supplies the unit
        of the axes' data coordinates through ImageCoordinates.
        """
        if image is not None:
            ax, _ = plot_image(image, ax=ax, overlays=self, **plot_options)
            return ax
        if ax is None or not ax.images or plot_options:
            raise ValueError("Supply an image, or pass axes with an existing image and no plot options.")
        existing = getattr(ax.images[-1], "_radio_diffusion_image_coordinates", None)
        if existing is not None and existing.unit != self.coordinates.unit:
            raise ValueError(f"Existing image axes are in {existing.unit}, not {self.coordinates.unit}.")
        self._draw(ax)
        return ax

    def _draw(self, ax):
        for item in self._items:
            item.draw(ax)
        if any(item.label for item in self._items):
            ax.legend()

    @staticmethod
    def _from_artist(
        description, factory, *args, label=None, color="cyan", linewidth=1.5,
        linestyle="solid", alpha=None, **options,
    ):
        make_artist = partial(
            factory, *args, label=label, color=color, linewidth=linewidth,
            linestyle=linestyle, alpha=alpha, **options,
        )
        return _OverlayItem(description, lambda ax: ax.add_artist(make_artist()), label)

    @classmethod
    def arrow(cls, start, end, **style):
        """Point from start to end."""
        start, end = _overlay_points((start, end))
        return cls._from_artist(
            f"Arrow from {_point_text(start)} to {_point_text(end)}",
            FancyArrowPatch, start, end, arrowstyle="->", mutation_scale=12,
            shrinkA=0, shrinkB=0, **style,
        )

    @classmethod
    def line(cls, start, end, **style):
        """Connect two points with a straight segment."""
        points = _overlay_points((start, end))
        return cls._from_artist(
            f"Line from {_point_text(points[0])} to {_point_text(points[1])}",
            Line2D, *points.T, **style,
        )

    @classmethod
    def ellipse(cls, center, radius_x, radius_y, *, angle_deg=0.0, **style):
        """Draw an outline rotated from positive x toward positive y. Equal radii give a circle."""
        if (
            np.shape(center) != (2,) or not np.isfinite(center).all()
            or not np.isfinite([radius_x, radius_y, angle_deg]).all()
            or min(radius_x, radius_y) <= 0
        ):
            raise ValueError("Ellipse geometry must be finite with two positive radii.")
        return cls._from_artist(
            f"Ellipse at {_point_text(center)} with radii {radius_x:g}, {radius_y:g} and angle {angle_deg:g}°",
            Ellipse, tuple(center), width=2 * radius_x, height=2 * radius_y,
            angle=angle_deg, fill=False, **style,
        )

    @classmethod
    def arc(
        cls, center, radius, *, angle_deg, delta_angle_deg, fade=False, linestyle="solid", **style,
    ):
        """Draw a circular arc centered on angle_deg, with delta_angle_deg on each side.

        The angle is measured from positive x toward positive y in axes coordinates.
        With fade=True, a continuous stroke tapers from its alpha at the center to zero at the ends.
        """
        if (
            np.shape(center) != (2,) or not np.isfinite(center).all()
            or not np.isfinite([radius, angle_deg, delta_angle_deg]).all()
            or radius <= 0 or not 0 < delta_angle_deg <= 180
        ):
            raise ValueError("An arc needs a finite center and angle, a positive radius and 0 < delta_angle_deg <= 180.")
        description = (
            f"Arc at {_point_text(center)} with radius {radius:g}, "
            f"angle {angle_deg:g}° ± {delta_angle_deg:g}°"
        )
        if fade:
            if linestyle not in ("solid", "-"):
                raise ValueError("A faded arc requires a solid linestyle.")
            angles = np.deg2rad(angle_deg + np.linspace(-delta_angle_deg, delta_angle_deg, 98))
            points = np.asarray(center) + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            return cls._from_artist(
                f"{description}, faded", _faded_arc_artist, segments, linestyle=linestyle, **style,
            )
        return cls._from_artist(
            description,
            Arc, tuple(center), width=2 * radius, height=2 * radius,
            angle=angle_deg, theta1=-delta_angle_deg, theta2=delta_angle_deg, linestyle=linestyle, **style,
        )

    @classmethod
    def points(cls, points, *, marker=".", markersize=4, **style):
        """Draw a marker at each supplied (x, y) point."""
        points = _overlay_points(points, min_points=1)
        return cls._from_artist(
            f"Markers at {len(points)} {'point' if len(points) == 1 else 'points'}",
            Line2D, *points.T, marker=marker, markersize=markersize, linestyle="none", **style,
        )

    @classmethod
    def polyline(cls, points, *, smooth=False, **style):
        """Connect points in order, optionally interpolating a curve through them.

        Smoothing uses SciPy's PCHIP (piecewise cubic Hermite interpolation)
        with cumulative point-to-point distance as the curve parameter.
        """
        points = _overlay_points(points)
        description = f"{'Smoothed polyline' if smooth else 'Polyline'} through {len(points)} points"
        if smooth:
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            keep = np.r_[True, lengths > 0]
            if keep.sum() >= 3:
                distance = np.r_[0.0, np.cumsum(lengths[lengths > 0])]
                samples = np.linspace(distance[:-1], distance[1:], 12, endpoint=False, axis=-1)
                samples = np.append(samples.ravel(), distance[-1])
                points = PchipInterpolator(distance, points[keep], axis=0)(samples)
        return cls._from_artist(description, Line2D, *points.T, **style)


def _image_style(style, cmap, scale):
    if isinstance(style, str):
        if style not in IMAGE_STYLES:
            raise ValueError(f"Unknown image style {style!r}.")
        style = IMAGE_STYLES[style]
    return replace(
        style, cmap=style.cmap if cmap is None else cmap,
        scale=style.scale if scale is None else scale,
    )


def _image_array(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()
    array = np.ma.masked_invalid(np.ma.asarray(image))
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not array.size:
        raise ValueError("An image must have shape [H, W] or [1, H, W].")
    return array


def _intensity_norm(images, style, limits):
    if limits is None:
        values = []
        for image in images:
            finite = image.compressed()
            if style.scale == "log":
                finite = finite[finite > 0]
            if finite.size:
                values.append((float(finite.min()), float(finite.max())))
        if not values:
            qualifier = "positive finite" if style.scale == "log" else "finite"
            raise ValueError(f"Automatic color limits require at least one {qualifier} pixel.")
        low = min(pair[0] for pair in values)
        high = max(pair[1] for pair in values)
        if low == high:
            high = low * 2 if style.scale == "log" else low + max(abs(low) * 0.01, 1)
    else:
        low, high = limits
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("Color limits must be finite and increasing.")
    if style.scale == "log":
        if low <= 0:
            raise ValueError("Log color limits must be positive.")
        return LogNorm(vmin=low, vmax=high)
    if style.scale == "gamma":
        return PowerNorm(gamma=style.gamma, vmin=low, vmax=high)
    return Normalize(vmin=low, vmax=high)


def _image_transform(coordinates, shape):
    reference, matrix = coordinates._calibration()
    affine = np.eye(3)
    affine[:2, :2] = matrix
    affine[:2, 2] = -matrix @ reference
    transform = Affine2D(affine)
    height, width = shape
    corners = transform.transform([
        (-0.5, -0.5), (width - 0.5, -0.5),
        (-0.5, height - 0.5), (width - 0.5, height - 0.5),
    ])
    x, y = corners.T
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())
    return (
        transform, xlim[::-1] if coordinates.x_increases_left else xlim,
        ylim[::-1] if coordinates.y_increases_down else ylim,
    )


def _resolve_coordinates(coordinates, overlays):
    if coordinates is not None and not isinstance(coordinates, ImageCoordinates):
        raise TypeError("coordinates must be ImageCoordinates.")
    if isinstance(overlays, ImageOverlay):
        if coordinates is not None and coordinates != overlays.coordinates:
            raise ValueError("The image and its overlay collection must use the same coordinates.")
        return overlays.coordinates
    return _PIXEL_COORDINATES if coordinates is None else coordinates


def _add_colorbar(figure, artist, **options):
    bar = figure.colorbar(artist, **options)
    if artist.norm.clip:
        # A clipping stretch maps out-of-range ticks onto the endpoints.
        ticks = bar.get_ticks()
        inside = ticks[(bar.vmin <= ticks) & (ticks <= bar.vmax)]
        if inside.size != ticks.size:
            bar.set_ticks(inside if inside.size else [bar.vmin, bar.vmax])
    return bar


def plot_image(
    image, *, ax=None, style="default", cmap=None, scale=None, limits=None,
    norm=None, coordinates=None, show_axes=True, overlays=None, colorbar=False, colorbar_label="",
):
    """Draw an image and return its axes and artist. Overlays are optional.

    Default pixel axes use array indices with row zero at the top. Calibrated
    axes use the provided ImageCoordinates or the overlay collection's coordinates.
    Display normalization leaves the image values and their units intact.
    """
    coordinates = _resolve_coordinates(coordinates, overlays)
    if overlays is not None and not isinstance(overlays, ImageOverlay):
        items = overlays
        overlays = ImageOverlay(coordinates)
        for item in items:
            overlays.append(item)
    style = _image_style(style, cmap, scale)
    image = _image_array(image)
    transform, xlim, ylim = _image_transform(coordinates, image.shape)
    if norm is None:
        norm = _intensity_norm([image], style, limits)
    if isinstance(norm, LogNorm):
        image = np.ma.masked_less_equal(image, 0)
    if ax is None:
        ax = Figure(layout="constrained").subplots()

    artist = ax.imshow(
        image, origin="lower", transform=transform + ax.transData,
        cmap=style.cmap, norm=norm, interpolation="nearest", aspect="equal",
    )
    unit_label = "pixel" if coordinates.unit == "pixels" else f"({coordinates.unit})"
    ax.set(
        xlim=xlim, ylim=ylim, xlabel=f"{coordinates.x_label} {unit_label}",
        ylabel=f"{coordinates.y_label} {unit_label}",
    )
    artist._radio_diffusion_image_coordinates = coordinates
    if not show_axes:
        ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
    if overlays is not None:
        overlays._draw(ax)
    if colorbar:
        _add_colorbar(ax.figure, artist, ax=ax, label=colorbar_label)
    return ax, artist


def plot_grid(
    images, *, nrow=4, title=None, sample_labels=None, row_labels=None,
    column_labels=None, style="default", cmap=None, scale=None, limits=None,
    shared_scale=True, colorbar=True, colorbar_label="", coordinates=None,
    show_axes=None, overlays=None,
):
    """Draw a row-major image grid and return its figure and axes array.

    Coordinates may be shared or supplied per tile. Overlays are supplied as
    one collection or list of shapes per tile.
    """
    images = [_image_array(image) for image in images]
    if not images or nrow < 1:
        raise ValueError("A grid needs at least one image and a positive column count.")
    columns = min(nrow, len(images))
    rows = (len(images) + columns - 1) // columns
    per_tile_coordinates = isinstance(coordinates, (list, tuple))
    for name, items, count in (
        ("sample_labels", sample_labels, len(images)), ("row_labels", row_labels, rows),
        ("column_labels", column_labels, columns), ("overlays", overlays, len(images)),
        ("coordinates", coordinates if per_tile_coordinates else None, len(images)),
    ):
        if items is not None and len(items) != count:
            raise ValueError(f"{name} must contain {count} entries.")

    style = _image_style(style, cmap, scale)
    norm = _intensity_norm(images, style, limits) if shared_scale else None
    label_width = 2.5 if row_labels is not None else 0
    figure = Figure(
        figsize=(2.5 * columns + 1 + label_width, 2.5 * rows + 0.5), layout="constrained",
    )
    axes = figure.subplots(rows, columns, squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(images):
            ax.axis("off")
            continue
        panel_overlays = None if overlays is None else overlays[index]
        panel_coordinates = _resolve_coordinates(
            coordinates[index] if per_tile_coordinates else coordinates, panel_overlays,
        )
        panel_show_axes = panel_coordinates.unit != "pixels" if show_axes is None else show_axes
        _, artist = plot_image(
            images[index], ax=ax, style=style, norm=norm, limits=limits,
            coordinates=panel_coordinates, show_axes=panel_show_axes, overlays=panel_overlays,
            colorbar=colorbar and not shared_scale, colorbar_label=colorbar_label,
        )
        row, column = divmod(index, columns)
        if sample_labels is not None:
            ax.set_title(sample_labels[index], fontsize=8)
        elif column_labels is not None and row == 0:
            ax.set_title(column_labels[column], fontsize=9)
        if row_labels is not None and column == 0:
            ax.set_ylabel(
                row_labels[row] + (f"\n{ax.get_ylabel()}" if panel_show_axes else ""),
                rotation=0, ha="right", va="center", labelpad=12, fontsize=8,
            )
    if title is not None:
        figure.suptitle(title)
    if colorbar and shared_scale:
        _add_colorbar(figure, artist, ax=list(axes.flat), shrink=0.8, label=colorbar_label)
    return figure, axes


def save_sample_grid(images, path, *, dpi=120, **plot_options):
    """Save a grid using the same renderer as interactive plots."""
    figure, _ = plot_grid(images, **plot_options)
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi)
    finally:
        figure.clear()


#### Mojave specific plotting ###


_BEAM_PADDING_FRACTION = 0.035


def mojave_coordinates(metadata):
    """Adapt saved MOJAVE spatial_geometry to east-left offsets in mas."""
    return ImageCoordinates(
        unit="mas", reference_pixel_xy=metadata["reference_pixel_xy"],
        pixel_to_units=metadata["pixel_to_east_north_mas"],
        x_label="East offset", y_label="North offset", x_increases_left=True,
    )


def draw_restoring_beam(
    ax, major_mas, minor_mas, pa_deg, *, color="0.6", alpha=None,
    padding_fraction=_BEAM_PADDING_FRACTION,
):
    """Anchor a filled FWHM ellipse to the lower-left corner of mas axes.

    The two full widths are in mas, and PA is measured north through east.
    Padding is measured from the ellipse's outer edge as a fraction of each axes dimension.
    """
    if not np.isfinite([major_mas, minor_mas, pa_deg]).all() or not 0 < minor_mas <= major_mas:
        raise ValueError("Beam values must be finite with 0 < minor <= major.")
    if not 0 <= padding_fraction < 1:
        raise ValueError("padding_fraction must be at least zero and less than one.")

    beam = AuxTransformBox(ax.transData)
    beam.add_artist(Ellipse(
        (0, 0), width=major_mas, height=minor_mas, angle=90 - pa_deg,
        facecolor=color, edgecolor="none", alpha=alpha,
    ))
    marker = AnchoredOffsetbox(
        loc="lower left", child=beam, frameon=False, pad=0, borderpad=0,
        bbox_to_anchor=(padding_fraction, padding_fraction), bbox_transform=ax.transAxes,
    )
    return ax.add_artist(marker)


def restoring_beam(
    major_mas, minor_mas, pa_deg, *, color="0.6", alpha=None,
    padding_fraction=_BEAM_PADDING_FRACTION,
):
    """Prepare a restoring-beam marker to append to a mas overlay collection."""
    draw = partial(
        draw_restoring_beam, major_mas=major_mas, minor_mas=minor_mas, pa_deg=pa_deg, color=color,
        alpha=alpha, padding_fraction=padding_fraction,
    )
    return _OverlayItem(
        f"Restoring beam {major_mas:g} × {minor_mas:g} mas, PA {pa_deg:g}°",
        draw, required_unit="mas",
    )
