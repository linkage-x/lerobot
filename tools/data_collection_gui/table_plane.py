"""The mapping between one camera's pixels and the table plane in the robot's base frame.

Why this exists: the scene-reset map and the landing map are drawn in base x/y, and the camera
still behind them used to be stretched to fill the plot box. Stretching is not a projection --
the peg in the image lands wherever the aspect ratio happens to put it, centimetres away from
where the same peg is plotted, and an operator painting a target region against that backdrop
is aiming at the wrong part of the table.

One plane, one homography. A camera looking at a flat table sees a projective transform of it,
so four points whose base x/y is known recover the whole plane -- and the robot supplies those
points itself: drive the tool to a known base x/y at the plane height, take a still, click the
tool in it. That is the whole calibration, and it needs no board, no intrinsics and no
hand-eye solve.

Nothing here holds off that plane. A point 5 cm above the table reprojects wrong by roughly
its height times the camera's obliquity, which is why the plane height is stored with the fit,
reported with it, and why points taken at two different heights are refused rather than
averaged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

# Four is the minimum a plane homography has -- eight degrees of freedom, two equations per
# point. It is also the number that leaves no residual to look at: any four non-degenerate
# points fit exactly, so a four-point calibration reports 0.0 mm whether the operator clicked
# well or badly. Five is the first count that can be wrong out loud.
MIN_POINTS = 4
RECOMMENDED_POINTS = 5
# Points are a plane fit, so they all have to come off the same plane. 2 mm is well under the
# 6 mm the reset's own step tolerance allows, so a point that passes this was probed at the
# height it claims.
PLANE_Z_TOLERANCE_M = 0.002
# Below these the point set is a line rather than a quadrilateral, and the fit is a division by
# very nearly zero: it "succeeds" and then maps most of the image to infinity.
_MIN_BASE_SPREAD_M = 0.01
_MIN_IMAGE_SPREAD_PX = 5.0
# Four points fit a plane only if they are in general position -- no three of them on one
# straight line. Three collinear points state the same line twice and leave two of the eight
# degrees of freedom unconstrained, so the solve "succeeds" and returns a matrix that sends much
# of the image past the horizon. An operator walks straight into this: probe the four corners of
# a rectangle, find one unreachable, and substitute the centre -- which lies on both diagonals.
# The tolerances are the resolution the two halves of a point are known to anyway: 5 mm is the
# probe's own arrival tolerance, 5 px is about how well a fingertip can be clicked.
_MIN_TRIPLE_SEPARATION_M = 0.005
_MIN_TRIPLE_SEPARATION_PX = 5.0


class TablePlaneError(ValueError):
    """A calibration that cannot be fitted, or a point that must not join one."""


@dataclass(frozen=True)
class TablePlanePoint:
    """One correspondence: where the tool was, and where it appeared."""

    u: float
    v: float
    x: float
    y: float

    def payload(self) -> dict[str, float]:
        return {"u": self.u, "v": self.v, "x": self.x, "y": self.y}


@dataclass(frozen=True)
class TableWindow:
    """The base-frame rectangle a warped still is rendered into.

    The caller is a plot that already knows what its axes span; it asks for exactly that
    rectangle, so alignment between the image and the points drawn over it is a property of
    the request rather than something the two ends have to agree on separately.
    """

    minX: float
    maxX: float
    minY: float
    maxY: float

    def __post_init__(self) -> None:
        for name, value in (("minX", self.minX), ("maxX", self.maxX), ("minY", self.minY), ("maxY", self.maxY)):
            if not math.isfinite(value):
                raise TablePlaneError(f"window.{name} must be finite.")
        if self.maxX - self.minX <= 0.0 or self.maxY - self.minY <= 0.0:
            raise TablePlaneError("window must have positive width and height.")

    def payload(self) -> dict[str, float]:
        return {"minX": self.minX, "maxX": self.maxX, "minY": self.minY, "maxY": self.maxY}


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TablePlaneError(f"{field_name} must be a number, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TablePlaneError(f"{field_name} must be a number.") from exc
    if not math.isfinite(parsed):
        raise TablePlaneError(f"{field_name} must be finite.")
    return parsed


def _matrix_from_payload(raw: Any, field_name: str) -> list[list[float]] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise TablePlaneError(f"{field_name} must be a 3x3 matrix.")
    matrix: list[list[float]] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise TablePlaneError(f"{field_name}[{row_index}] must have three entries.")
        matrix.append([_finite(value, f"{field_name}[{row_index}][{index}]") for index, value in enumerate(row)])
    return matrix


def _spread(values: Sequence[Sequence[float]]) -> float:
    """The narrower principal standard deviation of a 2D point set.

    Collinear points have zero spread across the line, which is exactly the case a homography
    cannot be fitted from and the case an operator falls into by probing four points along one
    edge of the table.
    """

    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] < 2:
        return 0.0
    centred = array - array.mean(axis=0)
    # Singular values of the centred matrix are the principal spreads, scaled by sqrt(n).
    singular = np.linalg.svd(centred, compute_uv=False)
    return float(singular[-1] / math.sqrt(array.shape[0]))


def _normalization_matrix(points: Any) -> Any:
    """Hartley normalization: centre on the mean, scale the mean distance to sqrt(2).

    Without it the DLT matrix mixes columns of order 1 with columns of order 640*0.5, and the
    smallest singular vector is chosen by rounding error rather than by the data.
    """

    import numpy as np

    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(np.sqrt((shifted ** 2).sum(axis=1)).mean())
    scale = math.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0
    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _triple_deviation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """How far the middle point of a triple is off the line through the other two.

    The triangle's shortest altitude, which is the distance that matters regardless of which of
    the three is the odd one out.
    """

    area_twice = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    longest = max(math.dist(a, b), math.dist(a, c), math.dist(b, c))
    return area_twice / longest if longest > 0.0 else 0.0


def _require_general_position(points: Sequence[TablePlanePoint]) -> None:
    """Refuse a point set in which no four points form a proper quadrilateral.

    More than four points survive a collinear triple -- the extra point still constrains the
    fit, and the centre of a well-probed rectangle is a perfectly good fifth measurement even
    though it sits on both diagonals. What cannot be tolerated is having no clean four anywhere
    in the set, because then nothing pins the mapping down.
    """

    lines: list[tuple[int, int, int, float, float]] = []
    for i, j, k in combinations(range(len(points)), 3):
        a, b, c = points[i], points[j], points[k]
        base_off = _triple_deviation((a.x, a.y), (b.x, b.y), (c.x, c.y))
        image_off = _triple_deviation((a.u, a.v), (b.u, b.v), (c.u, c.v))
        if base_off < _MIN_TRIPLE_SEPARATION_M or image_off < _MIN_TRIPLE_SEPARATION_PX:
            lines.append((i, j, k, base_off, image_off))
    if not lines:
        return
    straight = [frozenset(line[:3]) for line in lines]
    for quad in combinations(range(len(points)), 4):
        if not any(triple <= set(quad) for triple in straight):
            return

    i, j, k, base_off, image_off = min(lines, key=lambda line: line[3])
    raise TablePlaneError(
        f"Points {i + 1}, {j + 1} and {k + 1} lie on one straight line -- the middle one is "
        f"{base_off * 1000:.0f} mm off it on the table and {image_off:.0f} px off it in the "
        "image. A plane fit needs four points with no three in a row, and the centre of a "
        "rectangle is on both of its diagonals, so it cannot stand in for a corner the arm "
        "cannot reach. Probe a point off that line instead -- half way out towards one edge, "
        "away from the centre."
    )


def fit_image_to_base(points: Sequence[TablePlanePoint]) -> tuple[list[list[float]], list[float]]:
    """Solve the plane homography that takes image pixels to base metres.

    Returns the matrix and the per-point residual in millimetres -- the distance between where
    the robot actually was and where the fit says the click was. That number is the whole
    quality report: it is in the units the operator's error budget is written in, and it is the
    only thing that distinguishes a good calibration from four points clicked carelessly.
    """

    import numpy as np

    if len(points) < MIN_POINTS:
        raise TablePlaneError(f"A plane fit needs at least {MIN_POINTS} points; {len(points)} recorded.")
    image = np.array([[point.u, point.v] for point in points], dtype=np.float64)
    base = np.array([[point.x, point.y] for point in points], dtype=np.float64)
    if _spread(image) < _MIN_IMAGE_SPREAD_PX or _spread(base) < _MIN_BASE_SPREAD_M:
        raise TablePlaneError(
            "The points are too close to a straight line to define a plane. Spread them out "
            "over the working area -- four corners of a rectangle is the shape to aim for."
        )
    _require_general_position(points)

    t_image = _normalization_matrix(image)
    t_base = _normalization_matrix(base)
    image_h = np.hstack([image, np.ones((len(points), 1))]) @ t_image.T
    base_h = np.hstack([base, np.ones((len(points), 1))]) @ t_base.T

    rows: list[list[float]] = []
    for (u, v, _w), (x, y, w) in zip(image_h, base_h, strict=True):
        rows.append([0.0, 0.0, 0.0, -w * u, -w * v, -w, y * u, y * v, y])
        rows.append([w * u, w * v, w, 0.0, 0.0, 0.0, -x * u, -x * v, -x])
    _u, _s, vt = np.linalg.svd(np.array(rows, dtype=np.float64))
    normalized = vt[-1].reshape(3, 3)
    matrix = np.linalg.inv(t_base) @ normalized @ t_image
    if not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        raise TablePlaneError("The points do not define a usable plane mapping; clear them and probe again.")
    matrix = matrix / matrix[2, 2]

    residuals: list[float] = []
    for point in points:
        x, y = project_image_to_base([[float(value) for value in row] for row in matrix], point.u, point.v)
        residuals.append(round(math.hypot(x - point.x, y - point.y) * 1000.0, 2))
    return [[float(value) for value in row] for row in matrix], residuals


def project_image_to_base(matrix: Sequence[Sequence[float]], u: float, v: float) -> tuple[float, float]:
    denominator = matrix[2][0] * u + matrix[2][1] * v + matrix[2][2]
    if abs(denominator) < 1e-12:
        raise TablePlaneError("This pixel projects to the plane's horizon; it cannot be a table point.")
    x = (matrix[0][0] * u + matrix[0][1] * v + matrix[0][2]) / denominator
    y = (matrix[1][0] * u + matrix[1][1] * v + matrix[1][2]) / denominator
    return x, y


def invert(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    import numpy as np

    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    if not np.isfinite(inverse).all() or abs(inverse[2, 2]) < 1e-12:
        raise TablePlaneError("The plane mapping is not invertible.")
    inverse = inverse / inverse[2, 2]
    return [[float(value) for value in row] for row in inverse]


def base_to_window_matrix(window: TableWindow, width: int, height: int) -> list[list[float]]:
    """Base metres to output pixels for one render window.

    Base +x is drawn up the screen and base +y to the left, matching both maps: that is the
    view an operator has standing in front of the cell, and the axis convention is fixed here
    so the served image and the SVG drawn over it cannot drift apart.
    """

    if width <= 0 or height <= 0:
        raise TablePlaneError("window render size must be positive.")
    scale_x = width / (window.maxX - window.minX)
    scale_y = height / (window.maxY - window.minY)
    # Pixel centres, not corners: pixel (0, 0) covers [minX, minX + 1/scale), whose centre is
    # half a pixel in. Getting this wrong offsets the whole backdrop by half a pixel, which is
    # invisible and permanent.
    return [
        [scale_x, 0.0, -scale_x * window.minX - 0.5],
        [0.0, -scale_y, scale_y * window.maxY - 0.5],
        [0.0, 0.0, 1.0],
    ]


def warp_still_to_window(
    jpeg: bytes,
    *,
    image_to_base: Sequence[Sequence[float]],
    window: TableWindow,
    width: int,
    height: int,
    source_size: tuple[int, int] | None = None,
    quality: int = 82,
) -> bytes:
    """Render one camera still as the plot's own rectangle of table.

    Done here rather than with a CSS transform in the browser because the plot's axes are the
    contract: the server is told the exact base-frame rectangle the caller is drawing and
    returns that rectangle, so a stale bundle cannot half-apply a transform and produce a
    backdrop that is subtly wrong instead of visibly absent.
    """

    import numpy as np

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - the rig always has it; CI may not
        raise TablePlaneError(f"OpenCV is needed to warp the camera still: {exc}") from exc

    source = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise TablePlaneError("The camera still could not be decoded.")
    # The homography is in the pixels of the frames the calibration was clicked on. A still of a
    # different size is the same scene through a different lens as far as this matrix is
    # concerned, and warping it anyway would produce a backdrop that is plausibly placed and
    # wrong by a scale factor -- the exact failure the whole projection replaced.
    if source_size is not None and (source.shape[1], source.shape[0]) != tuple(source_size):
        raise TablePlaneError(
            f"This still is {source.shape[1]}x{source.shape[0]} and the alignment was measured "
            f"on {source_size[0]}x{source_size[1]} frames. Re-align the camera, or serve the "
            "still from the process the alignment was made against."
        )
    transform = np.asarray(base_to_window_matrix(window, width, height), dtype=np.float64) @ np.asarray(
        image_to_base, dtype=np.float64
    )
    warped = cv2.warpPerspective(
        source,
        transform,
        (int(width), int(height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        # Table that the camera cannot see is left black rather than smeared with edge pixels,
        # so "the camera does not cover this corner" reads as a hole instead of as scenery.
        borderValue=(0, 0, 0),
    )
    ok, buffer = cv2.imencode(".jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise TablePlaneError("The warped still could not be encoded.")
    return buffer.tobytes()


@dataclass
class TablePlaneCalibration:
    """Everything one camera's table mapping is made of, and where it came from."""

    cameraKey: str
    planeZ: float
    imageWidth: int = 0
    imageHeight: int = 0
    points: list[TablePlanePoint] = field(default_factory=list)
    imageToBase: list[list[float]] | None = None
    residualsMm: list[float] = field(default_factory=list)
    # Why enough points produced no mapping. Kept with the points because the operator reads it
    # long after the click that caused it -- and because a delete that leaves the set degenerate
    # has to report the state it left behind rather than refusing to happen.
    fitError: str = ""
    updatedAt: str = ""

    @property
    def calibrated(self) -> bool:
        return self.imageToBase is not None and len(self.points) >= MIN_POINTS

    @property
    def maxResidualMm(self) -> float:
        return max(self.residualsMm) if self.residualsMm else 0.0

    def add_point(self, point: TablePlanePoint, *, plane_z: float) -> None:
        if abs(plane_z - self.planeZ) > PLANE_Z_TOLERANCE_M and self.points:
            raise TablePlaneError(
                f"The stored points were probed at z={self.planeZ:.3f} m and this one is at "
                f"z={plane_z:.3f} m. A homography maps one plane; clear the points to "
                "recalibrate at a new height."
            )
        self.planeZ = plane_z
        self.points.append(point)
        self.refit()

    def refit(self, *, strict: bool = True) -> None:
        """Recompute the mapping, or leave the points uncalibrated with a reason.

        A point set below the minimum is not an error state -- it is a calibration in progress
        -- so this clears the matrix instead of raising. A set that is large enough but
        degenerate raises when `strict`: the operator asked for a fit and has to hear why there
        is none. Deleting a point is the exception. Removing one from a good set can leave four
        that no longer form a quadrilateral, and refusing the delete would trap the operator in
        the set they were trying to escape -- so that path records the reason and carries on.
        """

        self.updatedAt = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if len(self.points) < MIN_POINTS:
            self.imageToBase = None
            self.residualsMm = []
            self.fitError = ""
            return
        try:
            self.imageToBase, self.residualsMm = fit_image_to_base(self.points)
        except TablePlaneError as exc:
            self.imageToBase = None
            self.residualsMm = []
            self.fitError = str(exc)
            if strict:
                raise
            return
        self.fitError = ""

    def payload(self) -> dict[str, Any]:
        return {
            "cameraKey": self.cameraKey,
            "planeZ": round(self.planeZ, 5),
            "imageWidth": int(self.imageWidth),
            "imageHeight": int(self.imageHeight),
            "points": [point.payload() for point in self.points],
            "imageToBase": self.imageToBase,
            "residualsMm": list(self.residualsMm),
            "maxResidualMm": round(self.maxResidualMm, 2),
            "fitError": self.fitError,
            "calibrated": self.calibrated,
            "minPoints": MIN_POINTS,
            "recommendedPoints": RECOMMENDED_POINTS,
            "updatedAt": self.updatedAt,
        }


def calibration_from_payload(raw: Any, *, camera_key: str = "") -> TablePlaneCalibration:
    if not isinstance(raw, dict):
        raise TablePlaneError("table plane calibration must be a JSON object.")
    points_raw = raw.get("points")
    points: list[TablePlanePoint] = []
    if isinstance(points_raw, list):
        for index, item in enumerate(points_raw):
            if not isinstance(item, dict):
                raise TablePlaneError(f"points[{index}] must be an object.")
            points.append(
                TablePlanePoint(
                    u=_finite(item.get("u"), f"points[{index}].u"),
                    v=_finite(item.get("v"), f"points[{index}].v"),
                    x=_finite(item.get("x"), f"points[{index}].x"),
                    y=_finite(item.get("y"), f"points[{index}].y"),
                )
            )
    return TablePlaneCalibration(
        cameraKey=str(raw.get("cameraKey") or camera_key),
        planeZ=_finite(raw.get("planeZ", 0.0), "planeZ"),
        imageWidth=int(raw.get("imageWidth") or 0),
        imageHeight=int(raw.get("imageHeight") or 0),
        points=points,
        imageToBase=_matrix_from_payload(raw.get("imageToBase"), "imageToBase"),
        residualsMm=[_finite(value, "residualsMm") for value in raw.get("residualsMm") or []],
        fitError=str(raw.get("fitError") or ""),
        updatedAt=str(raw.get("updatedAt") or ""),
    )


def load_calibration(path: Path, *, camera_key: str) -> TablePlaneCalibration:
    """Read a stored calibration, or an empty one if this camera has never been aligned."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return TablePlaneCalibration(cameraKey=camera_key, planeZ=0.0)
    try:
        return calibration_from_payload(raw, camera_key=camera_key)
    except TablePlaneError:
        # A file this process cannot read is a file an operator has to be able to overwrite by
        # probing again, so it degrades to "not calibrated" rather than failing every request
        # that touches the map.
        return TablePlaneCalibration(cameraKey=camera_key, planeZ=0.0)


def save_calibration(path: Path, calibration: TablePlaneCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(calibration.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
