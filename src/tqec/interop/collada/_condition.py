"""Geometry helpers for the visible links from conditions to conditional cubes."""

from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from tqec.utils.position import SignedDirection3D

_TOLERANCE = 1e-9


def _surface_anchor(
    patches: Sequence[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float64] | None:
    """Project the area-weighted center of unique rectangles onto a real patch."""
    rectangles = {}
    for matrix in patches:
        origin = matrix[:3, 3].astype(np.float64)
        vectors = matrix[:3, :2].astype(np.float64)
        area = float(np.linalg.norm(np.cross(vectors[:, 0], vectors[:, 1])))
        if area <= _TOLERANCE:
            continue
        corners = (
            origin,
            origin + vectors[:, 0],
            origin + vectors[:, 1],
            origin + vectors.sum(axis=1),
        )
        key = tuple(sorted(tuple(np.round(corner, decimals=9)) for corner in corners))
        rectangles[key] = (origin, vectors, area)
    if not rectangles:
        return None

    # Sort by physical corners so ties do not depend on condition-edge iteration order.
    pieces = [rectangles[key] for key in sorted(rectangles)]
    centroid = sum(
        (origin + vectors.sum(axis=1) / 2) * area for origin, vectors, area in pieces
    ) / sum(area for _, _, area in pieces)
    candidates = []
    for origin, vectors, _ in pieces:
        coordinates = (centroid - origin) @ vectors / np.sum(vectors * vectors, axis=0)
        candidates.append(origin + vectors @ np.clip(coordinates, 0, 1))
    return min(candidates, key=lambda point: float(np.linalg.norm(point - centroid)))


def _distinct_points(points: Sequence[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """Remove consecutive repeated points without changing the path's endpoints."""
    distinct = []
    for point in points:
        if not distinct or np.linalg.norm(point - distinct[-1]) > _TOLERANCE:
            distinct.append(point)
    return np.asarray(distinct, dtype=np.float64).reshape(-1, 3)


def _round_corners(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Round short sections of interior corners, preserving the exterior bridge."""
    rounded = [points[0]]
    for previous, corner, following in zip(points, points[1:], points[2:]):
        incoming = corner - previous
        outgoing = following - corner
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        cut = min(0.2, incoming_length / 4, outgoing_length / 4)
        start = corner - incoming * (cut / incoming_length)
        end = corner + outgoing * (cut / outgoing_length)
        rounded.extend(
            (1 - t) ** 2 * start + 2 * (1 - t) * t * corner + t**2 * end
            for t in np.linspace(0, 1, 9)
        )
    rounded.append(points[-1])
    return _distinct_points(rounded)


def get_condition_link_path(
    patches: Sequence[npt.NDArray[np.float32]],
    target_center: npt.NDArray[np.float32],
    free_direction: SignedDirection3D,
    bounds: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Route a condition link around the structure and onto the cube's free cap.

    ``patches`` contains affine transforms of unit XY rectangles, and ``bounds``
    contains the minimum and maximum world coordinates of the structure. Choose
    the shortest of four side routes perpendicular to the free-cap direction.
    Each route has a bridge 0.6 units outside the structure; corner cuts of at
    most 0.2 units retain its exposed section. Ties prefer negative directions,
    with Y before X before Z. Return an empty path when there are no real patches.

    """
    source = _surface_anchor(patches)
    if source is None:
        return np.empty((0, 3), dtype=np.float64)
    free_axis = free_direction.direction.value
    outward = np.zeros(3, dtype=np.float64)
    outward[free_axis] = 1 if free_direction.towards_positive else -1
    cap = target_center.astype(np.float64) + 0.5 * outward
    stub = cap + 0.6 * outward

    shortest_path = None
    shortest_length = float("inf")
    for positive in (False, True):
        for axis in (1, 0, 2):
            if axis == free_axis:
                continue
            side = bounds[int(positive), axis] + (0.6 if positive else -0.6)
            projected_source, projected_stub = source.copy(), stub.copy()
            projected_source[axis] = projected_stub[axis] = side
            path = _distinct_points([source, projected_source, projected_stub, stub, cap])
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if length < shortest_length - _TOLERANCE:
                shortest_path, shortest_length = path, length
    assert shortest_path is not None
    return _round_corners(shortest_path)


def sample_dashes(
    path: npt.NDArray[np.float64], dash_length: float = 0.15, gap: float = 0.10
) -> Iterator[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Yield straight dash pieces with continuous spacing along a sampled path.

    Anchor the pattern at the target so its final dash reaches the cube. Split
    dashes at path vertices to follow rounded corners without restarting the
    pattern. Returned pieces follow the source-to-target path order.

    """
    if dash_length <= 0 or gap < 0:
        raise ValueError("Dash length must be positive and gap must be nonnegative.")
    points = _distinct_points(list(path))
    if len(points) < 2:
        return
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    intervals = [
        (max(0.0, end - dash_length), end)
        for end in np.arange(cumulative[-1], 0, -(dash_length + gap))
    ]
    for dash_start, dash_end in reversed(intervals):
        segment = max(0, int(np.searchsorted(cumulative, dash_start, side="right")) - 1)
        while segment < len(lengths) and cumulative[segment] < dash_end:
            start = max(dash_start, cumulative[segment])
            end = min(dash_end, cumulative[segment + 1])
            if end - start > _TOLERANCE:
                vector = (points[segment + 1] - points[segment]) / lengths[segment]
                yield (
                    points[segment] + vector * (start - cumulative[segment]),
                    points[segment] + vector * (end - cumulative[segment]),
                )
            segment += 1
