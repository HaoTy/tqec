from itertools import permutations, product

import numpy as np
import pytest

from tqec.computation.cube import ConditionalCubeKind, LeafCubeKind, StaticCubeKind, ZXCube
from tqec.interop.collada._geometry import BlockGeometries, Face
from tqec.interop.color import TQECColor
from tqec.utils.position import Direction3D, SignedDirection3D

CONDITIONAL_KINDS = [
    ConditionalCubeKind((branch_0, branch_1))
    for branch_0, branch_1 in permutations([*ZXCube, LeafCubeKind.Y_HALF_CUBE], 2)
    if not (isinstance(branch_0, ZXCube) and isinstance(branch_1, ZXCube))
    or sum(a != b for a, b in zip(branch_0.value, branch_1.value)) == 1
]
SIGNED_DIRECTIONS = [
    SignedDirection3D(direction, towards_positive)
    for direction in Direction3D.all_directions()
    for towards_positive in (False, True)
]


def _color(kind: StaticCubeKind, direction: Direction3D) -> TQECColor:
    return (
        TQECColor(kind.get_basis_along(direction).value)
        if isinstance(kind, ZXCube)
        else TQECColor.Y
    )


@pytest.mark.parametrize("kind", CONDITIONAL_KINDS, ids=str)
def test_conditional_cube_exterior_halves(kind: ConditionalCubeKind) -> None:
    geometries = BlockGeometries()
    assert kind not in geometries.geometries
    faces = geometries.get_geometry(kind)
    assert kind in geometries.geometries
    assert geometries.get_geometry(kind) == faces
    assert kind.pipe_direction is not None
    split_axis = (kind.pipe_direction.value + 1) % 3

    for normal in SIGNED_DIRECTIONS:
        exterior = [face for face in faces if face.normal_direction == normal]
        assert sum(face.width * face.height for face in exterior) == 1.0
        colors = tuple(_color(branch, normal.direction) for branch in kind.branches)
        # Coincident branch colors stay merged, as do faces wholly in one half.
        assert len(exterior) == (
            1 if normal.direction.value == split_axis or colors[0] == colors[1] else 2
        )
        for face in exterior:
            vertices = face.get_vertices().reshape(4, 3)
            assert np.all((0 <= vertices) & (vertices <= 1))
            # No face is placed on the internal dividing plane.
            np.testing.assert_array_equal(
                vertices[:, normal.direction.value], float(normal.towards_positive)
            )
            center = vertices.mean(axis=0)
            branch_index = int(center[split_axis] > 0.5)
            assert face.color == colors[branch_index]
            if colors[0] != colors[1]:
                if branch_index == 0:
                    assert vertices[:, split_axis].max() <= 0.5
                else:
                    assert vertices[:, split_axis].min() >= 0.5


@pytest.mark.parametrize("kind", CONDITIONAL_KINDS, ids=str)
@pytest.mark.parametrize("towards_positive", (False, True))
def test_conditional_cube_open_pipe_cap(kind: ConditionalCubeKind, towards_positive: bool) -> None:
    assert kind.pipe_direction is not None
    pipe_cap = SignedDirection3D(kind.pipe_direction, towards_positive)
    geometries = BlockGeometries()
    faces = geometries.get_geometry(kind, frozenset({pipe_cap}))
    assert all(face.normal_direction != pipe_cap for face in faces)
    free_cap = [face for face in faces if face.normal_direction == -pipe_cap]
    assert len(free_cap) == 2
    split_axis = (kind.pipe_direction.value + 1) % 3
    for branch_index, face in enumerate(free_cap):
        assert face.color == _color(kind.branches[branch_index], kind.pipe_direction)
        assert face.width * face.height == 0.5
        vertices = face.get_vertices().reshape(4, 3)
        assert vertices[:, split_axis].min() == branch_index * 0.5
        assert vertices[:, split_axis].max() == (branch_index + 1) * 0.5
    assert free_cap[0].color != free_cap[1].color
    assert all(
        face.normal_direction.direction != kind.pipe_direction
        for face in geometries.get_geometry(kind, frozenset({pipe_cap, -pipe_cap}))
    )


@pytest.mark.parametrize("kind", CONDITIONAL_KINDS, ids=str)
@pytest.mark.parametrize("popped_direction", SIGNED_DIRECTIONS, ids=str)
def test_conditional_cube_pop_any_face(
    kind: ConditionalCubeKind, popped_direction: SignedDirection3D
) -> None:
    geometries = BlockGeometries()
    full_geometry = geometries.get_geometry(kind)
    assert geometries.get_geometry(kind, frozenset({popped_direction})) == [
        face for face in full_geometry if face.normal_direction != popped_direction
    ]
    # Removing faces from one instance must not change its cached geometry.
    assert geometries.get_geometry(kind) == full_geometry


def test_conditional_y_geometry_is_full_height() -> None:
    geometries = BlockGeometries()
    ordinary_y = geometries.get_geometry(LeafCubeKind.Y_HALF_CUBE)
    conditional_y = geometries.get_geometry(
        ConditionalCubeKind((ZXCube.ZXX, LeafCubeKind.Y_HALF_CUBE))
    )
    assert max(face.get_vertices().reshape(4, 3)[:, 2].max() for face in ordinary_y) == 0.5
    y_faces = [face for face in conditional_y if face.color == TQECColor.Y]
    assert max(face.get_vertices().reshape(4, 3)[:, 2].max() for face in y_faces) == 1.0


def _frame_rod_centers(faces: list[Face]) -> set[tuple[float, ...]]:
    centers: set[tuple[float, ...]] = set()
    assert len(faces) % 6 == 0
    for offset in range(0, len(faces), 6):
        rod = faces[offset : offset + 6]
        assert {face.normal_direction for face in rod} == set(SIGNED_DIRECTIONS)
        assert all(face.color == TQECColor.CONDITIONAL_FRAME for face in rod)
        vertices = np.concatenate([face.get_vertices().reshape(4, 3) for face in rod])
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        # The frame projects out from the surface and rods overlap at their joins.
        np.testing.assert_allclose(sorted(upper - lower), [0.03, 0.03, 1.03])
        assert np.all(lower >= -0.015)
        assert np.all(upper <= 1.015)
        center = tuple(np.round((lower + upper) / 2, decimals=8))
        assert center not in centers
        centers.add(center)
    return centers


@pytest.mark.parametrize("kind", CONDITIONAL_KINDS, ids=str)
def test_conditional_frame_only_exterior_edges(kind: ConditionalCubeKind) -> None:
    geometries = BlockGeometries()
    faces = geometries.get_conditional_frame_geometry(kind)
    expected = set()
    for line_axis in range(3):
        for first, second in product((0.0, 1.0), repeat=2):
            center = [first, second]
            center.insert(line_axis, 0.5)
            expected.add(tuple(center))
    # Exactly twelve exterior rods, with no frame rods dividing the face interiors.
    assert len(faces) == 12 * 6
    assert _frame_rod_centers(faces) == expected
    assert (
        geometries.get_conditional_frame_geometry(
            ConditionalCubeKind((kind.branches[1], kind.branches[0]))
        )
        == faces
    )


@pytest.mark.parametrize("kind", CONDITIONAL_KINDS, ids=str)
@pytest.mark.parametrize("removed", SIGNED_DIRECTIONS, ids=str)
def test_conditional_frame_removed_face(
    kind: ConditionalCubeKind, removed: SignedDirection3D
) -> None:
    geometries = BlockGeometries()
    full_frame = _frame_rod_centers(geometries.get_conditional_frame_geometry(kind))
    # Edges bordering the opening remain as the borders of neighboring faces.
    assert (
        _frame_rod_centers(geometries.get_conditional_frame_geometry(kind, frozenset({removed})))
        == full_frame
    )


@pytest.mark.parametrize("kind", CONDITIONAL_KINDS, ids=str)
@pytest.mark.parametrize("retained", SIGNED_DIRECTIONS, ids=str)
def test_conditional_frame_single_retained_face(
    kind: ConditionalCubeKind, retained: SignedDirection3D
) -> None:
    geometries = BlockGeometries()
    removed = frozenset(set(SIGNED_DIRECTIONS) - {retained})
    centers = _frame_rod_centers(geometries.get_conditional_frame_geometry(kind, removed))
    assert len(centers) == 4
    assert all(
        center[retained.direction.value] == float(retained.towards_positive) for center in centers
    )


def test_conditional_frame_all_faces_removed() -> None:
    assert not BlockGeometries().get_conditional_frame_geometry(
        CONDITIONAL_KINDS[0], frozenset(SIGNED_DIRECTIONS)
    )
