from io import BytesIO

import collada
import numpy as np
import pytest

from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.interop.collada._correlation import CorrelationSurfaceTransformationHelper
from tqec.interop.collada.read_write import (
    _BaseColladaData,
    _Transformation,
    write_block_graph_to_dae_file,
)
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Direction3D, Position3D


def _surface(u: Position3D, v: Position3D, basis: Basis = Basis.X) -> CorrelationSurface:
    return CorrelationSurface(frozenset({ZXEdge(ZXNode(u, basis), ZXNode(v, basis))}))


def _vertices(transformation: _Transformation) -> np.ndarray:
    vertices = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]])
    return (vertices @ transformation.to_4d_affine_matrix().T)[:, :3]


@pytest.mark.parametrize("direction", Direction3D.all_directions())
@pytest.mark.parametrize("pipe_length", [0.5, 2.0, 5.0])
def test_partial_surface_open_endpoints(direction: Direction3D, pipe_length: float) -> None:
    graph = BlockGraph()
    u = Position3D(0, 0, 0)
    v = u.shift_in_direction(direction, 1)
    kind = "XZZ" if direction == Direction3D.Y else "ZXZ"
    graph.add_cube(u, kind)
    graph.add_cube(v, kind)
    graph.add_pipe(u, v)
    helper = CorrelationSurfaceTransformationHelper(graph, pipe_length)

    patches = helper.get_transformations_for_partial_surface(_surface(u, v))

    # A partial surface needs no even-parity completion at either endpoint.
    assert len(patches) == 3
    assert all(basis == Basis.X for basis, _ in patches)
    points = np.concatenate([_vertices(patch) for _, patch in patches])
    assert points[:, direction.value].min() == pytest.approx(0.5)
    assert points[:, direction.value].max() == pytest.approx(pipe_length + 1.5)
    endpoint_points = [_vertices(patch) for _, patch in patches[1:]]
    assert np.ptp(endpoint_points[0][:, direction.value]) == pytest.approx(0.5)
    assert np.ptp(endpoint_points[1][:, direction.value]) == pytest.approx(0.5)

    reversed_patches = helper.get_transformations_for_partial_surface(_surface(v, u))
    for (basis, patch), (reversed_basis, reversed_patch) in zip(patches, reversed_patches):
        assert basis == reversed_basis
        np.testing.assert_allclose(
            patch.to_4d_affine_matrix(), reversed_patch.to_4d_affine_matrix()
        )


def test_partial_surface_at_prior_conditional_cube() -> None:
    graph = BlockGraph()
    past, u, v = Position3D(0, 0, -1), Position3D(0, 0, 0), Position3D(0, 0, 1)
    graph.add_cube(past, "ZXZ")
    graph.add_cube(u, "ZXZ")
    graph.add_cube(v, "ZXZ_ZXX", condition=_surface(past, past))
    graph.add_pipe(u, v)
    helper = CorrelationSurfaceTransformationHelper(graph, 2.0)

    patches = helper.get_transformations_for_partial_surface(_surface(u, v))
    assert len(patches) == 3
    assert graph[v].is_conditional
    np.testing.assert_allclose(_vertices(patches[-1][1]).max(axis=0), [0.5, 1.0, 3.5])

    # Complete surfaces keep their existing static-cube rules, but conditional
    # endpoints use the same branch-independent half-cube sheet.
    complete_patches = helper.get_transformations_for_correlation_surface(_surface(u, v, Basis.Z))
    assert len(complete_patches) == 3
    np.testing.assert_allclose(_vertices(complete_patches[-1][1]).max(axis=0), [1.0, 0.5, 3.5])


@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
def test_partial_surface_hadamard_keeps_endpoint_planes_connected(basis: Basis) -> None:
    graph = BlockGraph()
    u, v = Position3D(0, 0, 0), Position3D(0, 0, 1)
    graph.add_cube(u, "ZXZ")
    graph.add_cube(v, "XZX")
    graph.add_pipe(u, v, "ZXOH")
    surface = CorrelationSurface(frozenset({ZXEdge(ZXNode(v, basis.flipped()), ZXNode(u, basis))}))

    patches = CorrelationSurfaceTransformationHelper(
        graph, 2.0
    ).get_transformations_for_partial_surface(surface)

    assert [patch_basis for patch_basis, _ in patches] == [
        basis,
        basis.flipped(),
        basis,
        basis.flipped(),
    ]
    normal_axis = 0 if basis == Basis.X else 1
    for _, patch in patches:
        np.testing.assert_allclose(_vertices(patch)[:, normal_axis], 0.5)
    assert _vertices(patches[0][1])[:, 2].max() == pytest.approx(2.0)
    assert _vertices(patches[1][1])[:, 2].min() == pytest.approx(2.0)


@pytest.mark.parametrize("endpoint_kind", ["P", "Y"])
def test_partial_surface_omits_port_and_y_interiors(endpoint_kind: str) -> None:
    graph = BlockGraph()
    u, v = Position3D(0, 0, 0), Position3D(0, 0, 1)
    graph.add_cube(u, "ZXZ")
    graph.add_cube(v, endpoint_kind, label="output")
    graph.add_pipe(u, v)

    patches = CorrelationSurfaceTransformationHelper(
        graph, 2.0
    ).get_transformations_for_partial_surface(_surface(u, v))

    assert len(patches) == 2
    assert max(_vertices(patch)[:, 2].max() for _, patch in patches) == pytest.approx(3.0)


def test_partial_surface_singleton_tiles() -> None:
    graph = BlockGraph()
    position = Position3D(1, 2, 3)
    graph.add_cube(position, "ZXZ")
    helper = CorrelationSurfaceTransformationHelper(graph, 2.0)

    patches = helper.get_transformations_for_partial_surface(_surface(position, position))
    assert len(patches) == 1
    assert patches[0][0] == Basis.X
    np.testing.assert_allclose(_vertices(patches[0][1]).mean(axis=0), [3.5, 6.5, 9.49])
    np.testing.assert_allclose(np.ptp(_vertices(patches[0][1]), axis=0), [0.3, 0.3, 0.0])

    both_bases = _surface(position, position, Basis.X) ^ _surface(position, position, Basis.Z)
    patches = helper.get_transformations_for_partial_surface(both_bases)
    assert [basis for basis, _ in patches] == [Basis.X, Basis.Z]
    assert patches[0][1].translation[2] < patches[1][1].translation[2]
    for basis, patch in patches:
        singleton = helper.get_transformations_for_partial_surface(
            _surface(position, position, basis)
        )
        np.testing.assert_array_equal(
            singleton[0][1].to_4d_affine_matrix(), patch.to_4d_affine_matrix()
        )


@pytest.mark.parametrize(
    ("kind", "pipe_z_offset", "expected_tile_z"),
    [("P", None, 0.49), ("Y", None, 0.24), ("Y", 1, 0.74), ("Y", -1, -0.26)],
)
def test_partial_singleton_on_port_and_y_cube(
    kind: str, pipe_z_offset: int | None, expected_tile_z: float
) -> None:
    graph = BlockGraph()
    position = Position3D(0, 0, 0)
    graph.add_cube(position, kind, label="singleton")
    if pipe_z_offset is not None:
        neighbor = position.shift_by(dz=pipe_z_offset)
        graph.add_cube(neighbor, "ZXZ")
        graph.add_pipe(position, neighbor)
    helper = CorrelationSurfaceTransformationHelper(graph, 2.0)

    patches = helper.get_transformations_for_partial_surface(_surface(position, position))

    assert len(patches) == 1
    np.testing.assert_allclose(_vertices(patches[0][1]).mean(axis=0), [0.5, 0.5, expected_tile_z])


def test_partial_surface_empty_span() -> None:
    helper = CorrelationSurfaceTransformationHelper(BlockGraph(), 2.0)
    assert helper.get_transformations_for_partial_surface(CorrelationSurface(frozenset())) == []


def test_partial_surface_missing_cube() -> None:
    graph = BlockGraph()
    u, v = Position3D(0, 0, 0), Position3D(0, 0, 1)
    graph.add_cube(u, "ZXZ")
    helper = CorrelationSurfaceTransformationHelper(graph, 2.0)

    with pytest.raises(TQECError, match=r"missing cube at \(0,0,1\)"):
        helper.get_transformations_for_partial_surface(_surface(u, v))


def test_partial_surface_missing_pipe() -> None:
    graph = BlockGraph()
    u, v = Position3D(0, 0, 0), Position3D(0, 0, 1)
    graph.add_cube(u, "ZXZ")
    graph.add_cube(v, "ZXZ")
    helper = CorrelationSurfaceTransformationHelper(graph, 2.0)

    with pytest.raises(TQECError, match=r"missing pipe between \(0,0,0\) and \(0,0,1\)"):
        helper.get_transformations_for_partial_surface(_surface(u, v))


def test_complete_overlay_replaces_contained_condition_patches() -> None:
    graph = BlockGraph()
    positions = [Position3D(0, 0, z) for z in range(3)]
    for position, kind in zip(positions, ("ZXX", "ZXZ", "ZXX")):
        graph.add_cube(position, kind)
    graph.add_pipe(positions[0], positions[1])
    graph.add_pipe(positions[1], positions[2])
    condition = _surface(positions[0], positions[1])
    complete = condition ^ _surface(positions[1], positions[2])
    neighbor, target = Position3D(1, 0, 2), Position3D(1, 0, 3)
    graph.add_cube(neighbor, "ZXZ")
    graph.add_cube(target, "ZXZ_ZXX", condition=condition)
    graph.add_pipe(neighbor, target)
    graph.validate()

    exports = []
    for show_conditions in (False, True):
        output = BytesIO()
        write_block_graph_to_dae_file(
            graph, output, show_correlation_surface=complete, show_conditions=show_conditions
        )
        mesh = collada.Collada(BytesIO(output.getvalue()))
        assert mesh.scene is not None
        patches = [
            tuple(node.matrix.flatten())
            for node in mesh.scene.nodes[0].children
            if node.name.endswith("correlation_surface")
        ]
        assert len(patches) == 5
        exports.append(set(patches))
    # Adding the condition must not darken the regions of the complete surface
    # that contain its half-cube endpoint patches.
    assert exports[0] == exports[1]


@pytest.mark.parametrize("larger_first", [False, True])
def test_surface_containment_deduplication_is_order_independent(larger_first: bool) -> None:
    base = _BaseColladaData()
    larger = _Transformation(
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.eye(3, dtype=np.float32),
    )
    smaller = _Transformation(
        np.array([0.25, 0.25, 0.0], dtype=np.float32),
        np.array([0.5, 0.5, 1.0], dtype=np.float32),
        np.eye(3, dtype=np.float32),
    )
    ordered = (larger, smaller) if larger_first else (smaller, larger)
    for patch in ordered:
        base.add_surface_instance(Basis.X, patch)

    assert len(base.root_node.children) == 1
    np.testing.assert_array_equal(base.root_node.children[0].matrix, larger.to_4d_affine_matrix())


@pytest.mark.parametrize("difference", ["basis", "plane", "normal", "partial", "rotation"])
def test_surface_containment_does_not_remove_distinct_patches(difference: str) -> None:
    base = _BaseColladaData()
    base.add_surface_instance(
        Basis.X,
        _Transformation(
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            np.eye(3, dtype=np.float32),
        ),
    )
    basis = Basis.X
    translation = np.array([0.25, 0.25, 0.0], dtype=np.float32)
    rotation = np.eye(3, dtype=np.float32)
    if difference == "basis":
        basis = Basis.Z
    elif difference == "plane":
        translation[2] = 0.1
    elif difference == "normal":
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    elif difference == "partial":
        translation[0] = 0.75
    else:
        sine = np.sqrt(0.5)
        translation[0] = 0.5
        rotation = np.array([[sine, -sine, 0], [sine, sine, 0], [0, 0, 1]], dtype=np.float32)
    base.add_surface_instance(
        basis, _Transformation(translation, np.array([0.5, 0.5, 1.0], dtype=np.float32), rotation)
    )

    assert len(base.root_node.children) == 2
