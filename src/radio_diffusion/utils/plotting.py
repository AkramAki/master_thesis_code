"""Matplotlib image plots, coordinate calibration, and annotations."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import numpy as np
import torch
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, PowerNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch
from matplotlib.transforms import Affine2D
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class ImageCoordinates:
    """Define the units and orientation of an image and its overlays.

    ``reference_pixel_xy`` is the ``(column, row)`` position on the image
    where ``(0, 0)`` in your overlay units lands.

    Pixel positions are ``(column, row)``, with ``(0, 0)`` at the center of
    the first pixel. ``ImageCoordinates()`` uses these positions directly,
    with x increasing right and y increasing down.

    For physical units, set ``reference_pixel_xy`` to the pixel position of
    the desired origin and provide an invertible 2 x 2 ``pixel_to_units``
    matrix. The mapping is ``pixel_to_units @ (pixel_xy - reference_pixel_xy)``.
    The matrix controls scale, sign, and rotation. ``x_increases_left`` and
    ``y_increases_down`` control the display direction of the calibrated axes.
    ``x_label`` and ``y_label`` name the axes, and ``unit`` is added to both.

    For example, this centers a plot on pixel ``(43.5, 43.5)`` with a scale
    of 0.1 mas per pixel and displays east to the left.

    >>> coordinates = ImageCoordinates(
    ...     unit="mas", reference_pixel_xy=(43.5, 43.5),
    ...     pixel_to_units=((-0.1, 0), (0, -0.1)),
    ...     x_label="East offset", y_label="North offset", x_increases_left=True,
    ... )

    Use ``from_pixels`` to convert positions already measured in pixels.
    Pass the coordinates to ``ImageOverlay`` and supply shape positions in
    the chosen units. To calibrate a plain image, call ``apply(ax)`` after
    drawing the image on ``ax`` with ``plot_image`` or ``ax.imshow``.
    """

    unit: str = "pixels"
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

    def from_pixels(self, pixel_xy: ArrayLike) -> np.ndarray:
        """Convert one or several (column, row) positions into axes coordinates."""
        return (np.asarray(pixel_xy) - self.reference_pixel_xy) @ np.asarray(self.pixel_to_units).T

    def apply(self, ax: Axes) -> Axes:
        """Calibrate the most recent image on ax and label the axes."""
        if not ax.images:
            raise ValueError("Draw an image on the axes before applying image coordinates.")
        artist = ax.images[-1]
        existing = getattr(artist, "_radio_diffusion_image_coordinates", None)
        if existing == self:
            return ax
        if existing is not None:
            raise ValueError("The image already uses a different coordinate system.")

        height, width = artist.get_array().shape[:2]
        left, right, bottom, top = artist.get_extent()
        row_start, row_end = (top, bottom) if artist.origin == "upper" else (bottom, top)
        if left == right or row_start == row_end:
            raise ValueError("Image calibration requires an extent with nonzero width and height.")

        # Map the artist's extent to pixel indices, including reversed extents.
        to_pixels = Affine2D().translate(-left, -row_start).scale(
            width / (right - left), height / (row_end - row_start),
        ).translate(-0.5, -0.5)
        matrix = np.asarray(self.pixel_to_units)
        affine = np.eye(3)
        affine[:2, :2] = matrix
        affine[:2, 2] = -matrix @ self.reference_pixel_xy
        to_units = Affine2D(affine)
        x, y = to_units.transform([
            (-0.5, -0.5), (width - 0.5, -0.5),
            (-0.5, height - 0.5), (width - 0.5, height - 0.5),
        ]).T
        xlim = (x.max(), x.min()) if self.x_increases_left else (x.min(), x.max())
        ylim = (y.max(), y.min()) if self.y_increases_down else (y.min(), y.max())

        artist.set_transform(to_pixels + to_units + ax.transData)
        unit_label = "pixel" if self.unit == "pixels" else f"({self.unit})"
        ax.set(
            xlim=xlim, ylim=ylim,
            xlabel=f"{self.x_label} {unit_label}", ylabel=f"{self.y_label} {unit_label}",
        )
        artist._radio_diffusion_image_coordinates = self
        return ax


def _faded_arc_artist(segments: ArrayLike, *, color, alpha: float | None, **style) -> LineCollection:
    colors = np.tile(to_rgba(color, alpha), (len(segments), 1))
    colors[:, 3] *= (1 + np.cos(np.linspace(-np.pi, np.pi, len(segments)))) / 2
    style.setdefault("zorder", 1)
    return LineCollection(segments, colors=colors, **style)


_GLYPH_ANGLE = np.linspace(0, 2 * np.pi, 33)
_GLYPH_ARC_X = np.linspace(0.1, 0.9, 21)
_GLYPH_PATHS = {
    "arrow": [(0.1, 0.5), (0.9, 0.5), (0.7, 0.85), (0.9, 0.5), (0.7, 0.15)],
    "line": [(0.1, 0.5), (0.9, 0.5)],
    "ellipse": np.column_stack((
        0.5 + 0.38 * np.cos(_GLYPH_ANGLE), 0.5 + 0.35 * np.sin(_GLYPH_ANGLE),
    )),
    "arc": np.column_stack((_GLYPH_ARC_X, 0.8 - 3.5 * (_GLYPH_ARC_X - 0.5) ** 2)),
    "points": [(0.35, 0.5), (0.65, 0.5)],
    "polyline": [(0.1, 0.2), (0.35, 0.75), (0.6, 0.35), (0.9, 0.7)],
}


@dataclass(frozen=True)
class OverlayGlyph:
    """Build a fresh Matplotlib artist in a unit-sized legend handle box."""

    make_artist: Callable[[], Artist]

    @classmethod
    def from_shape(
        cls, shape: str, *, color="cyan", alpha: float | None = None,
        fade: bool = False, **style,
    ) -> "OverlayGlyph":
        """Create and return a new glyph for the given shape.

        Call this method on the class, for example ``OverlayGlyph.from_shape("arrow")``.
        The glyph can also be used for labels without drawable geometry.
        """
        points = np.asarray(_GLYPH_PATHS[shape])
        if fade:
            segments = np.stack((points[:-1], points[1:]), axis=1)
            return cls(partial(_faded_arc_artist, segments, color=color, alpha=alpha, **style))
        if shape == "arrow" and style.get("linestyle") in ("dashed", "--"):
            shaft, tip = points[:2], points[2:]
            linestyles = [style.pop("linestyle"), "solid"]
            return cls(partial(
                LineCollection, [shaft, tip], colors=[color], alpha=alpha,
                linestyles=linestyles, **style,
            ))
        if shape == "points":
            style.setdefault("marker", ".")
            style.setdefault("linestyle", "none")
        return cls(partial(Line2D, *points.T, color=color, alpha=alpha, **style))


class _OverlayGlyphHandler(HandlerBase):
    def create_artists(self, legend, glyph, xdescent, ydescent, width, height, fontsize, trans):
        artist = glyph.make_artist()
        artist.set_transform(Affine2D().scale(width, height).translate(-xdescent, -ydescent) + trans)
        return [artist]


def overlay_legend(
    target: Axes | Figure, handles: Iterable[Artist | OverlayGlyph | tuple[Artist, ...]],
    labels: Iterable[str], *, handler_map=None, **options,
) -> Legend:
    """Draw a legend from caller-supplied handles and labels.

    The handles can include ``OverlayGlyph`` objects and native Matplotlib
    artists. This function registers the handler for overlay glyphs. Pass a
    ``Figure`` as the target to build a shared footer for a grid, or use
    ``ImageOverlay.legend`` for the labeled items of one collection.
    """
    handlers = {OverlayGlyph: _OverlayGlyphHandler(), **(handler_map or {})}
    return target.legend(handles=handles, labels=labels, handler_map=handlers, **options)


@dataclass(frozen=True)
class OverlayItem:
    description: str
    draw: Callable[[Axes], Artist | None]
    label: str | None = None
    glyph: OverlayGlyph | None = None
    show_glyph: bool = True


@dataclass
class ImageOverlay:
    """Collect shapes in one coordinate system and draw them in order.

    Shape geometry uses the collection's coordinates. Labels and glyphs are
    displayed when legend is called.
    """

    coordinates: ImageCoordinates = field(default_factory=ImageCoordinates)
    _items: list[OverlayItem] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.coordinates, ImageCoordinates):
            raise TypeError("ImageOverlay requires ImageCoordinates.")

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[OverlayItem]:
        return iter(self._items)

    def __getitem__(self, index: int | slice) -> OverlayItem | list[OverlayItem]:
        return self._items[index]

    def append(self, item: OverlayItem | list[OverlayItem] | tuple[OverlayItem, ...]) -> None:
        """Add one overlay item or a list or tuple of items.

        Create items with shape factories such as ``ImageOverlay.arrow``.
        Items are drawn in the order they are appended.
        """
        items = item if isinstance(item, (list, tuple)) else [item]
        if any(not isinstance(shape, OverlayItem) for shape in items):
            raise TypeError("Append an OverlayItem or a list of OverlayItems.")
        self._items.extend(items)

    def overlay(self, ax: Axes) -> Axes:
        """Transform the most recent image on ``ax`` and draw the overlay items.

        The collection's ``ImageCoordinates`` position and scale the image.
        Item positions use the units of those coordinates. Use the default
        ``ImageCoordinates()`` to keep standard image pixel positions. It maps
        each ``(column, row)`` to itself using an identity matrix and a
        reference position of ``(0, 0)``.
        """
        self.coordinates.apply(ax)
        for item in self._items:
            item.draw(ax)
        return ax

    def legend(self, ax: Axes, **options) -> Legend:
        """Draw a legend on ``ax`` from this collection's (this instance of ImageOverlay) labeled items.

        Each item's ``show_glyph`` controls whether its glyph is shown. For
        a shared figure footer, call ``overlay_legend`` with the figure and
        the handles and labels you want to display.
        """
        items = [item for item in self._items if item.label is not None]
        handles = [
            item.glyph if item.show_glyph and item.glyph is not None else Line2D([], [], visible=False)
            for item in items
        ]
        return overlay_legend(ax, handles, [item.label for item in items], **options)

    @staticmethod
    def _make_overlay_item(
        description: str, glyph_shape: str, artist_factory: Callable[..., Artist], *artist_args,
        add_to_axes: Callable[..., Artist],
        label: str | None = None, show_glyph: bool = True, fade_glyph: bool = False,
        color="cyan", linewidth: float = 1.5, linestyle="solid", alpha: float | None = None,
        **artist_options,
    ) -> OverlayItem:
        """Create an overlay item with a draw function and a legend glyph.

        Parameters
        ----------
        artist_factory
            Class or function that creates the Matplotlib artist, such as
            ``FancyArrowPatch`` for an arrow. Each draw calls this factory
            and adds the new artist to the target axes.
        *artist_args
            Positional arguments to the factory, such as the arrow endpoints.
        add_to_axes
            Axes method that attaches the artist, such as ``Axes.add_patch``.
        **artist_options
            Additional keyword arguments to the factory, such as ``arrowstyle``.
        """
        shared_style = dict(color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

        def draw(ax: Axes) -> Artist:
            artist = artist_factory(*artist_args, **shared_style, **artist_options)
            return add_to_axes(ax, artist)

        marker_style = {
            key: artist_options[key] for key in ("marker", "markersize") if key in artist_options
        }
        glyph = OverlayGlyph.from_shape(glyph_shape, fade=fade_glyph, **shared_style, **marker_style)
        return OverlayItem(
            description=description,
            draw=draw,
            label=label,
            glyph=glyph,
            show_glyph=show_glyph,
        )

    @classmethod
    def arrow(cls, start: ArrayLike, end: ArrayLike, **style) -> OverlayItem:
        """Point from start to end."""
        points = np.array((start, end), dtype=float, copy=True)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError("Arrow endpoints must be finite (x, y) pairs.")
        start, end = points
        return cls._make_overlay_item(
            f"Arrow from ({start[0]:g}, {start[1]:g}) to ({end[0]:g}, {end[1]:g})",
            "arrow", FancyArrowPatch, start, end, arrowstyle="->", mutation_scale=12,
            shrinkA=0, shrinkB=0, add_to_axes=Axes.add_patch, **style,
        )

    @classmethod
    def line(cls, start: ArrayLike, end: ArrayLike, **style) -> OverlayItem:
        """Connect two points with a straight segment."""
        points = np.array((start, end), dtype=float, copy=True)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError("Line endpoints must be finite (x, y) pairs.")
        start, end = points
        return cls._make_overlay_item(
            f"Line from ({start[0]:g}, {start[1]:g}) to ({end[0]:g}, {end[1]:g})",
            "line", Line2D, *points.T, add_to_axes=Axes.add_line, **style,
        )

    @classmethod
    def ellipse(
        cls, center: ArrayLike, radius_x: float, radius_y: float,
        *, angle_deg: float = 0.0, **style,
    ) -> OverlayItem:
        """Draw an outline rotated from positive x toward positive y. Equal radii give a circle."""
        if (
            np.shape(center) != (2,) or not np.isfinite(center).all()
            or not np.isfinite([radius_x, radius_y, angle_deg]).all()
            or min(radius_x, radius_y) <= 0
        ):
            raise ValueError("Ellipse geometry must be finite with two positive radii.")
        return cls._make_overlay_item(
            f"Ellipse at ({center[0]:g}, {center[1]:g}) "
            f"with radii {radius_x:g}, {radius_y:g} and angle {angle_deg:g}°",
            "ellipse", Ellipse, tuple(center), width=2 * radius_x, height=2 * radius_y,
            angle=angle_deg, fill=False, add_to_axes=Axes.add_patch, **style,
        )

    @classmethod
    def arc(
        cls, center: ArrayLike, radius: float, *, angle_deg: float, delta_angle_deg: float,
        fade: bool = False, linestyle="solid", **style,
    ) -> OverlayItem:
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
            f"Arc at ({center[0]:g}, {center[1]:g}) with radius {radius:g}, "
            f"angle {angle_deg:g}° ± {delta_angle_deg:g}°"
        )
        if fade:
            if linestyle not in ("solid", "-"):
                raise ValueError("A faded arc requires a solid linestyle.")
            angles = np.deg2rad(angle_deg + np.linspace(-delta_angle_deg, delta_angle_deg, 98))
            points = np.asarray(center) + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            return cls._make_overlay_item(
                f"{description}, faded", "arc", _faded_arc_artist, segments,
                fade_glyph=True, linestyle=linestyle, add_to_axes=Axes.add_collection, **style,
            )
        return cls._make_overlay_item(
            description,
            "arc", Arc, tuple(center), width=2 * radius, height=2 * radius,
            angle=angle_deg, theta1=-delta_angle_deg, theta2=delta_angle_deg,
            linestyle=linestyle, add_to_axes=Axes.add_patch, **style,
        )

    @classmethod
    def points(cls, points: ArrayLike, *, marker=".", markersize: float = 4, **style) -> OverlayItem:
        """Draw a marker at each supplied (x, y) point."""
        points = np.array(points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Overlay points must be finite (x, y) pairs.")
        return cls._make_overlay_item(
            f"Markers at {len(points)} {'point' if len(points) == 1 else 'points'}",
            "points", Line2D, *points.T, marker=marker, markersize=markersize,
            linestyle="none", add_to_axes=Axes.add_line, **style,
        )

    @classmethod
    def polyline(cls, points: ArrayLike, *, smooth: bool = False, **style) -> OverlayItem:
        """Connect points in order, optionally interpolating a curve through them.

        Smoothing uses SciPy's PCHIP (piecewise cubic Hermite interpolation)
        with cumulative point-to-point distance as the curve parameter.
        """
        points = np.array(points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Polyline points must be finite (x, y) pairs.")
        description = f"{'Smoothed polyline' if smooth else 'Polyline'} through {len(points)} points"
        if smooth:
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            keep = np.r_[True, lengths > 0]
            if keep.sum() >= 3:
                distance = np.r_[0.0, np.cumsum(lengths[lengths > 0])]
                samples = np.linspace(distance[:-1], distance[1:], 12, endpoint=False, axis=-1)
                samples = np.append(samples.ravel(), distance[-1])
                points = PchipInterpolator(distance, points[keep], axis=0)(samples)
        return cls._make_overlay_item(
            description, "polyline", Line2D, *points.T, add_to_axes=Axes.add_line, **style,
        )


def _as_2d_image_array(image: ArrayLike | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()
    array = np.asanyarray(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not array.size:
        raise ValueError("An image must have shape [H, W] or [1, H, W].")
    return array


def _set_default_image_norm(options):
    if "norm" not in options:
        options["norm"] = PowerNorm(
            gamma=0.5, vmin=options.pop("vmin", None), vmax=options.pop("vmax", None),
        )


def _prepare_shared_image_norm(images, options):
    _set_default_image_norm(options)
    norm = options["norm"]
    vmin, vmax = options.pop("vmin", None), options.pop("vmax", None)
    if isinstance(norm, Normalize):
        if vmin is not None or vmax is not None:
            raise ValueError("Set vmin and vmax on the supplied normalization instance.")
    else:
        mappable = ScalarMappable(norm=norm)
        mappable.set_clim(vmin, vmax)
        norm = mappable.norm
    if not norm.scaled():
        values = np.ma.concatenate([image.ravel() for image in images])
        norm.autoscale_None(np.ma.masked_invalid(values, copy=False))
    return norm


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
    image: ArrayLike | torch.Tensor, *, ax: Axes | None = None,
    show_axes: bool = True, colorbar: bool = False, colorbar_label: str = "",
    **imshow_options,
) -> tuple[Axes, AxesImage]:
    """Draw a scalar image and return its axes and imshow artist.

    Image options are forwarded to imshow. The default is an inferno colormap
    and a square-root stretch. Pass norm="linear" or norm="log" to change it.
    """
    _set_default_image_norm(imshow_options)
    imshow_options.setdefault("cmap", "inferno")
    imshow_options.setdefault("origin", "upper")
    imshow_options.setdefault("interpolation", "nearest")
    imshow_options.setdefault("aspect", "equal")
    if ax is None:
        ax = Figure(layout="constrained").subplots()

    artist = ax.imshow(_as_2d_image_array(image), **imshow_options)
    if not show_axes:
        ax.tick_params(
            which="both", bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False,
        )
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
    if colorbar:
        _add_colorbar(ax.figure, artist, ax=ax, label=colorbar_label)
    return ax, artist


def plot_grid(
    images: Iterable[ArrayLike | torch.Tensor], *, nrow: int = 4, title: str | None = None,
    sample_labels: Sequence[str] | None = None, row_labels: Sequence[str] | None = None,
    column_labels: Sequence[str] | None = None, figsize: tuple[float, float] | None = None,
    sharex: bool | Literal["none", "all", "row", "col"] = False,
    sharey: bool | Literal["none", "all", "row", "col"] = False, show_axes: bool = False,
    colorbar: Literal["shared", "each"] | None = "shared", colorbar_label: str = "",
    shared_scale: bool | None = None, **imshow_options,
) -> tuple[Figure, np.ndarray]:
    """Draw a row-major image grid and return its figure and axes array.

    A shared colorbar requires shared normalization. With individual colorbars
    or no colorbar, shared_scale can also request a common normalization.
    Axis sharing accepts the same options as Matplotlib subplots.
    """
    images = list(images)
    if not images or nrow < 1:
        raise ValueError("A grid needs at least one image and a positive column count.")
    if colorbar not in ("shared", "each", None):
        raise ValueError("Choose a shared colorbar, one for each panel, or None.")
    if shared_scale is None:
        shared_scale = colorbar == "shared"
    if colorbar == "shared" and not shared_scale:
        raise ValueError("A shared colorbar requires a shared scale.")

    columns = min(nrow, len(images))
    rows = (len(images) + columns - 1) // columns
    for name, items, count in (
        ("sample_labels", sample_labels, len(images)), ("row_labels", row_labels, rows),
        ("column_labels", column_labels, columns),
    ):
        if items is not None and len(items) != count:
            raise ValueError(f"{name} must contain {count} entries.")

    if shared_scale:
        images = [_as_2d_image_array(image) for image in images]
        imshow_options["norm"] = _prepare_shared_image_norm(images, imshow_options)
    if figsize is None:
        label_width = 2.5 if row_labels is not None else 0
        figsize = (2.5 * columns + 1 + label_width, 2.5 * rows + 0.5)
    figure = Figure(figsize=figsize, layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False, sharex=sharex, sharey=sharey)
    panels = list(axes.flat[:len(images)])
    for ax in axes.flat[len(images):]:
        ax.set_axis_off()
    for index, (ax, image) in enumerate(zip(panels, images)):
        plot_image(
            image, ax=ax, show_axes=show_axes, colorbar=colorbar == "each",
            colorbar_label=colorbar_label, **imshow_options,
        )
        row, column = divmod(index, columns)
        if sample_labels is not None:
            ax.set_title(sample_labels[index], fontsize=8)
        elif column_labels is not None and row == 0:
            ax.set_title(column_labels[column], fontsize=9)
        if row_labels is not None and column == 0:
            ax.set_ylabel(
                row_labels[row],
                rotation=0, ha="right", va="center", labelpad=12, fontsize=8,
            )
            ax.yaxis.label.set_visible(True)
    if title is not None:
        figure.suptitle(title)
    if colorbar == "shared":
        _add_colorbar(figure, panels[0].images[0], ax=panels, shrink=0.8, label=colorbar_label)
    return figure, axes


#### Mojave specific plotting ###


_BEAM_PADDING_FRACTION = 0.035


def mojave_coordinates(metadata) -> ImageCoordinates:
    """Adapt saved MOJAVE spatial_geometry to east-left offsets in mas."""
    return ImageCoordinates(
        unit="mas", reference_pixel_xy=metadata["reference_pixel_xy"],
        pixel_to_units=metadata["pixel_to_east_north_mas"],
        x_label="East offset", y_label="North offset", x_increases_left=True,
    )


def draw_restoring_beam(
    ax: Axes, major_mas: float, minor_mas: float, pa_deg: float, *, color="0.6",
    alpha: float | None = None, padding_fraction: float = _BEAM_PADDING_FRACTION,
) -> AnchoredOffsetbox:
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
    major_mas: float, minor_mas: float, pa_deg: float, *, color="0.6",
    alpha: float | None = None, label: str | None = None, show_glyph: bool = True,
    padding_fraction: float = _BEAM_PADDING_FRACTION,
) -> OverlayItem:
    """Prepare a restoring-beam marker to append to a mas overlay collection."""
    draw = partial(
        draw_restoring_beam, major_mas=major_mas, minor_mas=minor_mas, pa_deg=pa_deg, color=color,
        alpha=alpha, padding_fraction=padding_fraction,
    )
    return OverlayItem(
        f"Restoring beam {major_mas:g} × {minor_mas:g} mas, PA {pa_deg:g}°",
        draw, label, OverlayGlyph.from_shape("ellipse", color=color, alpha=alpha), show_glyph,
    )
