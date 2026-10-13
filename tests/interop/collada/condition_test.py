from itertools import pairwise

import numpy as np
import pytest

from tqec.interop.collada._condition import get_condition_link_path, sample_dashes
from tqec.utils.position import Direction3D, SignedDirection3D


def _patch(origin: tuple[float, float, float], width: float = 1, height: float = 1) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = origin
    matrix[0, 0], matrix[1, 1] = width, height
    return matrix


def _path(patches: list[np.ndarray]) -> np.ndarray:
    return get_condition_link_path(
        patches,
        np.array([4.5, 5.5, 6.5], dtype=np.float32),
        SignedDirection3D(Direction3D.Z, True),
        np.array([[0, 0, 0], [7, 7, 7]], dtype=np.float64),
    )


@pytest.mark.parametrize("axis", Direction3D.all_directions())
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_exterior_route_reaches_each_free_cap(
    axis: Direction3D, positive: bool, scale: float
) -> None:
    target = np.array([2, 3, 4], dtype=np.float32) * scale + 0.5
    bounds = np.array([[0, 0, 0], target + 0.5], dtype=np.float64)
    source = _patch((0, 0, 0.5))
    direction = SignedDirection3D(axis, positive)
    path = get_condition_link_path([source], target, direction, bounds)
    outward = np.zeros(3)
    outward[axis.value] = 1 if positive else -1
    np.testing.assert_allclose(path[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(path[-1], target + 0.5 * outward)
    np.testing.assert_allclose(
        (path[-1] - path[-2]) / np.linalg.norm(path[-1] - path[-2]), -outward
    )
    assert np.isfinite(path).all()
    assert (np.linalg.norm(np.diff(path, axis=0), axis=1) > 0).all()
    # A whole segment is beyond a side of the structure even after corner rounding.
    assert any(
        min(start[side], end[side]) > bounds[1, side] + 0.35
        or max(start[side], end[side]) < bounds[0, side] - 0.35
        for start, end in pairwise(path)
        for side in range(3)
        if side != axis.value
    )


def test_shortest_side_route_and_deterministic_ties() -> None:
    arguments = (
        [_patch((0, 0, 0.5))],
        np.array([0.5, 0.5, 3.5], dtype=np.float32),
        SignedDirection3D(Direction3D.Z, True),
    )
    symmetric_bounds = np.array([[0, 0, 0], [1, 1, 4]], dtype=np.float64)
    tied = get_condition_link_path(*arguments, symmetric_bounds)
    assert tied[:, 1].min() == pytest.approx(-0.6)
    assert tied[:, 0].min() == tied[:, 0].max() == 0.5
    # Moving the negative-Y boundary far away selects the shorter positive-Y side.
    asymmetric_bounds = np.array([[-10, -10, 0], [10, 1, 4]], dtype=np.float64)
    shorter = get_condition_link_path(*arguments, asymmetric_bounds)
    assert shorter[:, 1].max() == pytest.approx(1.6)


def test_source_uses_patch_area() -> None:
    path = _path([_patch((0, 0, 0)), _patch((3, 0, 0), width=3)])
    np.testing.assert_allclose(path[0], [3.5, 0.5, 0])


def test_source_stays_on_disconnected_surface_and_ignores_duplicate_patches() -> None:
    first, second = _patch((0, 0, 0)), _patch((3, 0, 0))
    swapped = first.copy()
    swapped[:3, :2] = first[:3, [1, 0]]
    original = _path([first, second])
    np.testing.assert_allclose(original[0], [1, 0.5, 0])
    np.testing.assert_allclose(_path([second, first, first, swapped]), original)


def test_source_stays_on_bent_surface() -> None:
    path = _path([_patch((0, 0, 0), width=4), _patch((0, 1, 0), height=3)])
    # The weighted center (19/14, 19/14, 0) is outside this L-shaped surface.
    np.testing.assert_allclose(path[0], [19 / 14, 1, 0])


def test_source_projects_onto_rotated_rectangles() -> None:
    first = _patch((0, 0, 0), width=2)
    second = _patch((0, 0, 0), width=2)
    second[:3, 1] = [0, 0, 1]
    path = _path([first, second])
    source = path[0]
    assert source[0] == pytest.approx(1)
    assert sorted(source[1:]) == pytest.approx([0, 0.25])


def test_empty_and_zero_area_patches_produce_no_route() -> None:
    assert _path([]).shape == (0, 3)
    assert _path([_patch((0, 0, 0), width=0)]).shape == (0, 3)
    real = _patch((0, 0, 0))
    np.testing.assert_allclose(_path([real, _patch((10, 10, 10), height=0)]), _path([real]))


def test_dashes_keep_phase_across_corners_and_end_at_target() -> None:
    path = np.array([[0, 0, 0], [0.23, 0, 0], [0.23, 0.5, 0]], dtype=np.float64)
    pieces = list(sample_dashes(path))

    def arclength(point: np.ndarray) -> float:
        return float(point[0] if point[1] == 0 else 0.23 + point[1])

    intervals = [(arclength(start), arclength(end)) for start, end in pieces]
    np.testing.assert_allclose(intervals, [[0.08, 0.23], [0.33, 0.48], [0.58, 0.73]])
    np.testing.assert_allclose(pieces[-1][1], path[-1])


def test_a_single_dash_can_continue_around_a_corner() -> None:
    path = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.2, 0]], dtype=np.float64)
    pieces = list(sample_dashes(path, dash_length=0.25, gap=0.1))
    assert len(pieces) == 2
    np.testing.assert_allclose(pieces[0][0], [0.05, 0, 0])
    np.testing.assert_allclose(pieces[0][1], path[1])
    np.testing.assert_allclose(pieces[1][0], path[1])
    np.testing.assert_allclose(pieces[1][1], path[-1])
    assert sum(np.linalg.norm(end - start) for start, end in pieces) == pytest.approx(0.25)


def test_dashes_do_not_restart_at_samples_or_repeated_points() -> None:
    path = np.column_stack((np.linspace(0, 1, 101), np.zeros((101, 2))))
    pieces = list(sample_dashes(np.repeat(path, 2, axis=0)))
    assert sum(np.linalg.norm(end - start) for start, end in pieces) == pytest.approx(0.6)
    for start, end in pieces:
        assert np.linalg.norm(end - start) > 0
        midpoint = (start[0] + end[0]) / 2
        assert (1 - midpoint) % 0.25 < 0.15
    np.testing.assert_allclose(pieces[-1][1], path[-1])


def test_rounded_route_preserves_dash_lengths() -> None:
    path = _path([_patch((0, 0, 0))])
    pieces = list(sample_dashes(path))
    lengths = []
    previous_end = None
    for start, end in pieces:
        length = float(np.linalg.norm(end - start))
        if previous_end is not None and np.allclose(previous_end, start):
            lengths[-1] += length
        else:
            lengths.append(length)
        previous_end = end
    assert 0 < lengths[0] <= 0.15 + 1e-9
    np.testing.assert_allclose(lengths[1:], 0.15)
    np.testing.assert_allclose(pieces[-1][1], path[-1])


@pytest.mark.parametrize("path", [[], [[1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
def test_degenerate_paths_have_no_dashes(path: list[list[float]]) -> None:
    assert list(sample_dashes(np.asarray(path, dtype=np.float64))) == []


def test_short_path_is_one_dash() -> None:
    path = np.array([[0, 0, 0], [0, 0, 0.05]], dtype=np.float64)
    pieces = list(sample_dashes(path))
    assert len(pieces) == 1
    np.testing.assert_allclose(pieces[0], path)


@pytest.mark.parametrize(("dash_length", "gap"), [(0, 0.1), (-1, 0.1), (0.15, -1)])
def test_invalid_dash_dimensions(dash_length: float, gap: float) -> None:
    with pytest.raises(ValueError, match="Dash length"):
        list(sample_dashes(np.zeros((2, 3)), dash_length=dash_length, gap=gap))
