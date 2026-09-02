import json
import math

import numpy as np
import pytest

from tools.data_collection_gui import table_plane


def _synthetic_camera():
    """A camera looking down at the table from the side, as a base-plane -> pixel homography.

    Built from an actual pose rather than from a made-up matrix so the test exercises the
    projective case the rig is in: a plane seen obliquely, where equal steps in base x are
    unequal steps in pixels and a stretched image can never line up.
    """

    # Camera 1.1 m in front of the table, 0.7 m up, pitched down 35 degrees, looking back at
    # the robot. Columns are the base axes expressed in camera axes.
    pitch = math.radians(-35.0)
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [-math.sin(pitch), 0.0, -math.cos(pitch)],
            [math.cos(pitch), 0.0, -math.sin(pitch)],
        ]
    )
    translation = rotation @ (-np.array([1.30, 0.0, 0.70]))
    intrinsics = np.array([[610.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])
    plane_z = 0.035
    # A plane homography keeps the two in-plane columns and folds the height into the offset.
    base_to_image = intrinsics @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], rotation[:, 2] * plane_z + translation]
    )
    return base_to_image / base_to_image[2, 2], plane_z


def _points_from(base_to_image, base_points):
    points = []
    for x, y in base_points:
        pixel = base_to_image @ np.array([x, y, 1.0])
        points.append(table_plane.TablePlanePoint(u=pixel[0] / pixel[2], v=pixel[1] / pixel[2], x=x, y=y))
    return points


def test_the_fit_recovers_the_camera_that_generated_the_points():
    base_to_image, _plane_z = _synthetic_camera()
    points = _points_from(base_to_image, [(0.37, -0.08), (0.37, 0.08), (0.53, 0.08), (0.53, -0.08)])

    matrix, residuals = table_plane.fit_image_to_base(points)

    assert max(residuals) < 0.01  # sub-10-micron: this is the maths, not the operator
    for point in points:
        x, y = table_plane.project_image_to_base(matrix, point.u, point.v)
        assert (x, y) == pytest.approx((point.x, point.y), abs=1e-6)


def test_a_point_clicked_wrong_shows_up_as_millimetres_of_residual():
    base_to_image, _plane_z = _synthetic_camera()
    points = _points_from(
        base_to_image, [(0.37, -0.08), (0.37, 0.08), (0.53, 0.08), (0.53, -0.08), (0.45, 0.0)]
    )
    # Five points, so the fit is over-determined and a bad click cannot hide inside it.
    misclicked = points[-1]
    points[-1] = table_plane.TablePlanePoint(u=misclicked.u + 14.0, v=misclicked.v, x=misclicked.x, y=misclicked.y)

    _matrix, residuals = table_plane.fit_image_to_base(points)

    assert max(residuals) > 2.0


def test_four_points_on_one_line_are_refused_rather_than_fitted():
    base_to_image, _plane_z = _synthetic_camera()
    points = _points_from(base_to_image, [(0.35, 0.0), (0.40, 0.0), (0.45, 0.0), (0.50, 0.0)])

    with pytest.raises(table_plane.TablePlaneError, match="straight line"):
        table_plane.fit_image_to_base(points)


def test_three_corners_plus_the_centre_are_refused_with_the_three_points_named():
    """The shape an operator lands in when one corner is out of the arm's reach.

    The centre of a rectangle is on both diagonals, so corner + opposite corner + centre is a
    straight line and no homography passes through it. The old code fitted it anyway and threw
    a horizon error from somewhere inside the residual loop, which told the operator nothing
    about which points to move.
    """

    base_to_image, _plane_z = _synthetic_camera()
    points = _points_from(base_to_image, [(0.44, -0.053), (0.44, -0.213), (0.28, -0.213), (0.36, -0.133)])

    with pytest.raises(table_plane.TablePlaneError) as excinfo:
        table_plane.fit_image_to_base(points)

    message = str(excinfo.value)
    assert "Points 1, 3 and 4" in message
    assert "straight line" in message
    assert "horizon" not in message


def test_a_point_half_way_out_replaces_the_unreachable_corner():
    base_to_image, _plane_z = _synthetic_camera()
    points = _points_from(base_to_image, [(0.44, -0.053), (0.44, -0.213), (0.28, -0.213), (0.40, -0.133)])

    matrix, residuals = table_plane.fit_image_to_base(points)

    assert max(residuals) < 0.01
    x, y = table_plane.project_image_to_base(matrix, points[0].u, points[0].v)
    assert (x, y) == pytest.approx((0.44, -0.053), abs=1e-6)


def test_the_centre_is_still_welcome_as_a_fifth_point():
    """One collinear triple does not spoil a set that has a clean quadrilateral in it."""

    base_to_image, _plane_z = _synthetic_camera()
    points = _points_from(
        base_to_image,
        [(0.44, -0.053), (0.44, -0.213), (0.28, -0.213), (0.28, -0.053), (0.36, -0.133)],
    )

    _matrix, residuals = table_plane.fit_image_to_base(points)

    assert len(residuals) == 5
    assert max(residuals) < 0.01


def test_deleting_a_point_into_a_degenerate_set_reports_instead_of_refusing():
    """Otherwise the delete that gets you out of a bad set is the one that is blocked."""

    base_to_image, plane_z = _synthetic_camera()
    points = _points_from(
        base_to_image,
        [(0.44, -0.053), (0.44, -0.213), (0.28, -0.213), (0.28, -0.053), (0.36, -0.133)],
    )
    calibration = table_plane.TablePlaneCalibration(cameraKey="side", planeZ=plane_z, points=list(points))
    calibration.refit()
    assert calibration.calibrated

    calibration.points.pop(3)  # the corner an operator would drop, leaving corner-corner-centre
    calibration.refit(strict=False)

    assert not calibration.calibrated
    assert "straight line" in calibration.fitError
    assert calibration.payload()["fitError"] == calibration.fitError

    calibration.points.pop()  # and the delete that follows clears the complaint
    calibration.refit(strict=False)
    assert calibration.fitError == ""


def test_a_fit_needs_four_points():
    base_to_image, _plane_z = _synthetic_camera()
    points = _points_from(base_to_image, [(0.37, -0.08), (0.37, 0.08), (0.53, 0.08)])

    with pytest.raises(table_plane.TablePlaneError, match="at least 4"):
        table_plane.fit_image_to_base(points)


def test_the_render_window_puts_base_x_up_and_base_y_left():
    window = table_plane.TableWindow(minX=0.30, maxX=0.60, minY=-0.15, maxY=0.15)
    matrix = np.asarray(table_plane.base_to_window_matrix(window, 300, 300))

    def to_pixel(x, y):
        pixel = matrix @ np.array([x, y, 1.0])
        return pixel[0] / pixel[2], pixel[1] / pixel[2]

    # The plot draws base x rightward on its own screen axes and base y upward, and the served
    # image has to use the same corners or the backdrop is mirrored.
    top_left = to_pixel(window.minX, window.maxY)
    bottom_right = to_pixel(window.maxX, window.minY)
    assert top_left == pytest.approx((-0.5, -0.5))
    assert bottom_right == pytest.approx((299.5, 299.5))


def test_a_point_probed_at_another_height_is_refused_not_averaged_in():
    calibration = table_plane.TablePlaneCalibration(cameraKey="side", planeZ=0.035)
    calibration.add_point(table_plane.TablePlanePoint(u=100.0, v=100.0, x=0.37, y=-0.08), plane_z=0.035)

    with pytest.raises(table_plane.TablePlaneError, match="one plane"):
        calibration.add_point(
            table_plane.TablePlanePoint(u=120.0, v=110.0, x=0.37, y=0.08), plane_z=0.075
        )


def test_a_calibration_survives_a_save_and_load_round_trip(tmp_path):
    base_to_image, plane_z = _synthetic_camera()
    calibration = table_plane.TablePlaneCalibration(
        cameraKey="side", planeZ=plane_z, imageWidth=640, imageHeight=480
    )
    for point in _points_from(base_to_image, [(0.37, -0.08), (0.37, 0.08), (0.53, 0.08), (0.53, -0.08)]):
        calibration.add_point(point, plane_z=plane_z)
    assert calibration.calibrated

    path = tmp_path / "side.json"
    table_plane.save_calibration(path, calibration)
    reloaded = table_plane.load_calibration(path, camera_key="side")

    assert reloaded.calibrated
    assert reloaded.planeZ == pytest.approx(plane_z)
    assert reloaded.imageToBase == calibration.imageToBase
    assert json.loads(path.read_text())["cameraKey"] == "side"


def test_an_unreadable_calibration_reads_as_not_calibrated(tmp_path):
    path = tmp_path / "side.json"
    path.write_text("{ this is not json")

    calibration = table_plane.load_calibration(path, camera_key="side")

    assert not calibration.calibrated
    assert calibration.points == []


def test_a_still_is_warped_into_the_window_the_caller_asked_for():
    cv2 = pytest.importorskip("cv2")
    base_to_image, plane_z = _synthetic_camera()
    points = _points_from(base_to_image, [(0.37, -0.08), (0.37, 0.08), (0.53, 0.08), (0.53, -0.08)])
    matrix, _residuals = table_plane.fit_image_to_base(points)

    # A white dot on the table at a known base point. After the warp it must land where the
    # plot would draw that point, which is the entire claim this endpoint makes.
    still = np.zeros((480, 640, 3), dtype=np.uint8)
    marker_base = (0.45, 0.02)
    marker_pixel = base_to_image @ np.array([marker_base[0], marker_base[1], 1.0])
    marker_pixel = marker_pixel[:2] / marker_pixel[2]
    cv2.circle(still, (int(round(marker_pixel[0])), int(round(marker_pixel[1]))), 5, (255, 255, 255), -1)
    ok, buffer = cv2.imencode(".jpg", still)
    assert ok

    window = table_plane.TableWindow(minX=0.30, maxX=0.60, minY=-0.15, maxY=0.15)
    warped_jpeg = table_plane.warp_still_to_window(
        buffer.tobytes(), image_to_base=matrix, window=window, width=300, height=300
    )
    warped = cv2.imdecode(np.frombuffer(warped_jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # Centroid of the bright pixels, not the brightest one: the dot warps into a blob several
    # pixels across and argmax would return whichever edge of it comes first in memory.
    rows, columns = np.nonzero(warped > 128)
    assert rows.size, "the marker disappeared in the warp"
    row, column = rows.mean(), columns.mean()
    expected = np.asarray(table_plane.base_to_window_matrix(window, 300, 300)) @ np.array(
        [marker_base[0], marker_base[1], 1.0]
    )
    assert (column, row) == pytest.approx((expected[0], expected[1]), abs=3.0)


def test_a_still_from_a_differently_sized_source_is_refused_rather_than_scaled():
    """The matrix is in the pixels it was measured in.

    A 1280x720 still warped with a homography clicked on 640x480 frames comes out looking like
    a table -- placed plausibly and wrong by a factor of two, which is worse than no backdrop.
    """
    cv2 = pytest.importorskip("cv2")
    base_to_image, _plane_z = _synthetic_camera()
    points = _points_from(base_to_image, [(0.37, -0.08), (0.37, 0.08), (0.53, 0.08), (0.53, -0.08)])
    matrix, _residuals = table_plane.fit_image_to_base(points)
    ok, buffer = cv2.imencode(".jpg", np.zeros((360, 480, 3), dtype=np.uint8))
    assert ok

    with pytest.raises(table_plane.TablePlaneError, match="480x360"):
        table_plane.warp_still_to_window(
            buffer.tobytes(),
            image_to_base=matrix,
            window=table_plane.TableWindow(minX=0.30, maxX=0.60, minY=-0.15, maxY=0.15),
            width=300,
            height=300,
            source_size=(640, 480),
        )
