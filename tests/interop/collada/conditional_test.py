import base64
import re
from io import BytesIO
from itertools import pairwise
from pathlib import Path

import collada
import numpy as np
import pytest

from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.interop.collada.read_write import write_block_graph_to_dae_file
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Direction3D, Position3D


def _surface(u: Position3D, v: Position3D, basis: Basis = Basis.X) -> CorrelationSurface:
    return CorrelationSurface(frozenset({ZXEdge(ZXNode(u, basis), ZXNode(v, basis))}))


def _graph(condition: CorrelationSurface | None = None, *, kind: str = "ZXZ_ZXX") -> BlockGraph:
    graph = BlockGraph()
    a, b, target = Position3D(0, 0, 0), Position3D(1, 0, 0), Position3D(1, 0, 1)
    graph.add_cube(a, "ZXZ")
    graph.add_cube(b, "ZXZ")
    graph.add_cube(target, kind, condition=_surface(a, b) if condition is None else condition)
    graph.add_pipe(a, b)
    graph.add_pipe(b, target)
    return graph


def _export(graph: BlockGraph, *, show_conditions: bool = True) -> collada.Collada:
    output = BytesIO()
    write_block_graph_to_dae_file(graph, output, show_conditions=show_conditions)
    output.seek(0)
    return collada.Collada(output)


def _instances(mesh: collada.Collada, suffix: str) -> list[collada.scene.Node]:
    assert mesh.scene is not None
    return [
        node for node in mesh.scene.nodes[0].children if node.children[0].node.name.endswith(suffix)
    ]


def _surface_signatures(mesh: collada.Collada) -> set[tuple]:
    return {
        (node.children[0].node.name, tuple(node.matrix.flatten()))
        for node in _instances(mesh, "_CORRELATION")
    }


def _world_vertices(node: collada.scene.Node) -> np.ndarray:
    vertices = np.concatenate(
        [
            primitive.vertex
            for geometry in node.children[0].node.children
            for primitive in geometry.geometry.primitives
        ]
    )
    return vertices @ node.matrix[:3, :3].T + node.matrix[:3, 3]


def _opacities(node: collada.scene.Node) -> set[float]:
    return {
        material.target.effect.diffuse[3]
        for geometry in node.children[0].node.children
        for material in geometry.materials
    }


def _cube_instance(mesh: collada.Collada, position: Position3D) -> collada.scene.Node:
    assert mesh.scene is not None
    return next(
        node
        for node in mesh.scene.nodes[0].children
        if np.allclose(node.matrix[:3, 3], np.array(position.as_tuple()) * 3)
        and not node.children[0].node.name.endswith(
            ("_CORRELATION", "_CONDITION_LINK", "_CONDITIONAL_FRAME")
        )
    )


@pytest.mark.parametrize(
    ("axis", "branches"),
    [
        (Direction3D.X, ("ZXZ", "XXZ")),
        (Direction3D.Y, ("XZZ", "XXZ")),
        (Direction3D.Z, ("ZXZ", "ZXX")),
    ],
)
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("remove_free_cap", [False, True])
def test_exported_conditional_caps_follow_attached_pipe(
    axis: Direction3D,
    branches: tuple[str, str],
    sign: int,
    reverse: bool,
    remove_free_cap: bool,
) -> None:
    graph = BlockGraph()
    target = Position3D(2, 2, 2)
    neighbor = target.shift_in_direction(axis, sign)
    kind = "_".join(reversed(branches) if reverse else branches)
    graph.add_cube(target, kind, condition=CorrelationSurface(frozenset()))
    graph.add_cube(neighbor, branches[0])
    graph.add_pipe(target, neighbor)
    output = BytesIO()
    free_direction = f"{'-' if sign > 0 else '+'}{axis}"
    write_block_graph_to_dae_file(
        graph,
        output,
        pop_faces_at_directions=[free_direction] if remove_free_cap else [],
        show_conditions=False,
    )
    output.seek(0)
    mesh = collada.Collada(output)
    node = _instances(mesh, kind)[0]
    attached_coordinate = 1 if sign > 0 else 0
    free_coordinate = 1 - attached_coordinate
    cap_materials = set()
    for geometry in node.children[0].node.children:
        vertices = np.concatenate([primitive.vertex for primitive in geometry.geometry.primitives])
        assert not np.allclose(vertices[:, axis.value], attached_coordinate)
        if np.allclose(vertices[:, axis.value], free_coordinate):
            cap_materials.add(geometry.materials[0].target.id)
    assert cap_materials == (set() if remove_free_cap else {"X_material", "Z_material"})
    frame = _instances(mesh, "_CONDITIONAL_FRAME")[0]
    frame_face_centers = np.array(
        [
            primitive.vertex.mean(axis=0)
            for geometry in frame.children[0].node.children
            for primitive in geometry.geometry.primitives
        ]
    )
    # Every rod lies along an exterior edge; no frame seam crosses a face interior.
    assert np.all(
        np.count_nonzero((frame_face_centers > 0.05) & (frame_face_centers < 0.95), axis=1) <= 1
    )


@pytest.mark.parametrize("kind", ["ZXZ_ZXX", "ZXX_ZXZ", "ZXX_Y", "Y_ZXX"])
@pytest.mark.parametrize("show_conditions", [False, True])
def test_conditional_frames_stay_visible_without_condition_overlays(
    kind: str, show_conditions: bool
) -> None:
    mesh = _export(_graph(kind=kind), show_conditions=show_conditions)
    frames = _instances(mesh, "_CONDITIONAL_FRAME")
    assert len(frames) == 1
    frame = frames[0]
    assert frame.name.endswith("conditional_frame")
    np.testing.assert_allclose(frame.matrix, _instances(mesh, kind)[0].matrix)
    for geometry in frame.children[0].node.children:
        for material in geometry.materials:
            np.testing.assert_allclose(material.target.effect.diffuse, (1, 1, 1, 1))
    vertices = _world_vertices(frame) - frame.matrix[:3, 3]
    assert np.all(vertices.min(axis=0) <= 0)
    assert np.all(vertices.max(axis=0) >= 1)
    assert np.all(vertices.min(axis=0) > -0.05)
    assert np.all(vertices.max(axis=0) < 1.05)


def test_conditional_frame_disappears_when_all_faces_are_removed() -> None:
    output = BytesIO()
    write_block_graph_to_dae_file(
        _graph(),
        output,
        pop_faces_at_directions=[f"{sign}{axis}" for sign in ("-", "+") for axis in "XYZ"],
        show_conditions=False,
    )
    output.seek(0)
    assert not _instances(collada.Collada(output), "_CONDITIONAL_FRAME")


@pytest.mark.parametrize("show_conditions", [False, True])
@pytest.mark.parametrize("method", ["dae", "html"])
@pytest.mark.parametrize(
    ("invalidity", "message"),
    [
        ("isolated", "exactly one pipe"),
        ("multiple", "exactly one pipe"),
        ("wrong_axis", "non-timelike"),
        ("walls", r"mismatched\s+colors"),
    ],
)
def test_invalid_conditional_cube_rejected_before_output(
    tmp_path: Path, show_conditions: bool, method: str, invalidity: str, message: str
) -> None:
    graph = BlockGraph()
    target = Position3D(0, 0, 1)
    graph.add_cube(target, "ZXZ_ZXX", condition=CorrelationSurface(frozenset()))
    if invalidity != "isolated":
        neighbor = Position3D(1, 0, 1) if invalidity == "wrong_axis" else Position3D(0, 0, 0)
        graph.add_cube(neighbor, "XZZ" if invalidity == "walls" else "ZXZ")
        pipe_kind = {"wrong_axis": "OXZ", "walls": "XZO"}.get(invalidity, "ZXO")
        graph.add_pipe(target, neighbor, pipe_kind)
    if invalidity == "multiple":
        graph.add_cube(Position3D(0, 0, 2), "ZXZ")
        graph.add_pipe(target, Position3D(0, 0, 2))
    output = tmp_path / f"invalid.{method}"
    with pytest.raises(TQECError, match=message):
        if method == "dae":
            graph.to_dae_file(output, show_conditions=show_conditions)
        else:
            graph.view_as_html(output, show_conditions=show_conditions)
    assert not output.exists()


@pytest.mark.parametrize("show_conditions", [False, True])
def test_conditions_control_annotations_and_block_opacity(show_conditions: bool) -> None:
    mesh = _export(_graph(), show_conditions=show_conditions)
    assert bool(_instances(mesh, "_CORRELATION")) == show_conditions
    links = _instances(mesh, "_CONDITION_LINK")
    assert bool(links) == show_conditions
    assert all(node.name.endswith("condition_link") for node in links)
    for effect in mesh.effects:
        if effect.id in {"X_effect", "Z_effect", "Y_effect", "H_effect"}:
            assert effect.diffuse[3] == 1.0
        if effect.id.endswith("_translucent"):
            assert show_conditions
            assert effect.diffuse[3] == pytest.approx(0.3)
        if effect.id in {"X_CORRELATION_effect", "Z_CORRELATION_effect"}:
            assert effect.diffuse[3] == pytest.approx(0.8)
    surface_opacity = 0.3 if show_conditions else 1.0
    for position in (Position3D(0, 0, 0), Position3D(1, 0, 0)):
        assert _opacities(_cube_instance(mesh, position)) == {surface_opacity}
    assert _opacities(_cube_instance(mesh, Position3D(1, 0, 1))) == {1.0}
    assert all(_opacities(node) == {surface_opacity} for node in _instances(mesh, "OXZ"))
    assert all(_opacities(node) == {1.0} for node in _instances(mesh, "ZXO"))
    for node in links:
        for geometry in node.children[0].node.children:
            for material in geometry.materials:
                assert material.target.effect.diffuse[3] == 1.0
                assert material.target.effect.diffuse[:3] == pytest.approx(
                    (material.target.effect.diffuse[0],) * 3
                )


def test_same_block_kinds_have_independent_opaque_and_translucent_instances() -> None:
    graph = _graph()
    a, b = Position3D(0, 2, 0), Position3D(1, 2, 0)
    graph.add_cube(a, "ZXZ")
    graph.add_cube(b, "ZXZ")
    graph.add_pipe(a, b)
    mesh = _export(graph)
    assert _opacities(_cube_instance(mesh, Position3D(0, 0, 0))) == {0.3}
    assert _opacities(_cube_instance(mesh, a)) == {1.0}
    pipes = _instances(mesh, "OXZ")
    assert len(pipes) == 2
    assert {_opacities(node).pop() for node in pipes} == {0.3, 1.0}
    assert len({node.children[0].node.id for node in pipes}) == 2


@pytest.mark.parametrize("kind", ["ZXZ_ZXX", "ZXX_ZXZ", "ZXX_Y", "Y_ZXX"])
def test_conditional_target_becomes_translucent_when_another_condition_crosses_it(
    kind: str,
) -> None:
    graph = _graph(kind=kind)
    earlier_target, later_target = Position3D(1, 0, 1), Position3D(2, 0, 2)
    neighbor = Position3D(2, 0, 1)
    graph.add_cube(neighbor, "ZXZ")
    graph.add_cube(later_target, "ZXZ_ZXX", condition=_surface(Position3D(1, 0, 0), earlier_target))
    graph.add_pipe(neighbor, later_target)
    mesh = _export(graph)
    assert _opacities(_cube_instance(mesh, earlier_target)) == {0.3}
    assert _opacities(_cube_instance(mesh, later_target)) == {1.0}
    assert _opacities(_cube_instance(mesh, neighbor)) == {1.0}
    frames = _instances(mesh, "_CONDITIONAL_FRAME")
    assert len(frames) == 2
    assert all(_opacities(frame) == {1.0} for frame in frames)


@pytest.mark.parametrize("endpoint_kind", ["Y", "PORT"])
def test_omitted_endpoint_sheets_do_not_make_endpoint_blocks_translucent(
    endpoint_kind: str,
) -> None:
    endpoint, neighbor = Position3D(3, 0, -1), Position3D(3, 0, 0)
    graph = _graph(_surface(endpoint, neighbor))
    graph.add_cube(endpoint, endpoint_kind, label="input")
    graph.add_cube(neighbor, "ZXZ")
    graph.add_pipe(endpoint, neighbor, "ZXO")
    mesh = _export(graph)
    assert _opacities(_cube_instance(mesh, neighbor)) == {0.3}
    assert all(_opacities(node) == {1.0} for node in _instances(mesh, "Y"))
    assert not _instances(mesh, "PORT")


@pytest.mark.parametrize("endpoint_kind", ["ZXZ", "Y", "PORT"])
def test_singleton_tiles_make_only_their_supporting_blocks_translucent(
    endpoint_kind: str,
) -> None:
    endpoint = Position3D(3, 0, 0)
    graph = _graph(_surface(endpoint, endpoint))
    graph.add_cube(endpoint, endpoint_kind, label="input")
    mesh = _export(graph)
    if endpoint_kind != "PORT":
        assert _opacities(_cube_instance(mesh, endpoint)) == {0.3}
    else:
        assert not _instances(mesh, "PORT")
    assert _opacities(_cube_instance(mesh, Position3D(0, 0, 0))) == {1.0}
    assert _opacities(_cube_instance(mesh, Position3D(1, 0, 1))) == {1.0}


def test_shared_condition_patches_are_deduplicated_but_targets_keep_links() -> None:
    graph = _graph()
    original = _export(graph)
    target = Position3D(0, 0, 1)
    graph.add_cube(target, "ZXZ_ZXX", condition=graph.conditional_cubes[0].condition)
    graph.add_pipe(Position3D(0, 0, 0), target)
    shared = _export(graph)
    signatures = _surface_signatures(shared)
    assert signatures == _surface_signatures(original)
    assert len(signatures) == len(_instances(shared, "_CORRELATION"))
    assert len(_instances(shared, "_CONDITION_LINK")) > len(_instances(original, "_CONDITION_LINK"))


@pytest.mark.parametrize("pipe_length", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("distant_geometry", [False, True])
def test_dashed_connector_routes_outside_model_to_free_cap(
    pipe_length: float, distant_geometry: bool
) -> None:
    graph = _graph()
    if distant_geometry:
        graph.add_cube(Position3D(-2, -2, -2), "ZXZ")
        graph.add_cube(Position3D(4, 3, 5), "ZXZ")
    output = BytesIO()
    write_block_graph_to_dae_file(graph, output, pipe_length)
    output.seek(0)
    mesh = collada.Collada(output)
    step = 1 + pipe_length
    source = np.array((step * 0.5 + 0.5, 0.5, 0.5))
    target = np.array((step + 0.5, 0.5, step + 1))
    segments = np.array(
        [
            np.array(((0.5, 0.5, 0), (0.5, 0.5, 1))) @ node.matrix[:3, :3].T + node.matrix[:3, 3]
            for node in _instances(mesh, "_CONDITION_LINK")
        ]
    )
    assert len(segments) > 1
    endpoints = segments.reshape((-1, 3))
    assert np.linalg.norm(endpoints - source, axis=1).min() < 0.25
    assert np.linalg.norm(endpoints - target, axis=1).min() < 1e-5
    origins = np.array([cube.position.as_tuple() for cube in graph.cubes]) * step
    lower, upper = origins.min(axis=0), origins.max(axis=0) + 1
    assert np.any((endpoints < lower - 0.1) | (endpoints > upper + 0.1))
    vectors = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / lengths[:, None]
    assert np.any(np.linalg.norm(np.cross(directions, directions[0]), axis=1) > 0.1)
    assert np.any(
        [
            np.linalg.norm(end - next_start) > 0.05
            for (_, end), (next_start, _) in pairwise(segments)
        ]
    )


@pytest.mark.parametrize(
    ("axis", "kind", "neighbor_kind"),
    [
        (Direction3D.X, "ZXZ_XXZ", "ZXZ"),
        (Direction3D.Y, "XZZ_XXZ", "XZZ"),
        (Direction3D.Z, "ZXZ_ZXX", "ZXZ"),
    ],
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_condition_link_attaches_to_cap_opposite_the_pipe(
    axis: Direction3D, kind: str, neighbor_kind: str, sign: int
) -> None:
    graph = BlockGraph()
    a, b, target = Position3D(0, 0, 0), Position3D(1, 0, 0), Position3D(2, 2, 2)
    neighbor = target.shift_in_direction(axis, sign)
    graph.add_cube(a, "ZXZ")
    graph.add_cube(b, "ZXZ")
    graph.add_pipe(a, b)
    graph.add_cube(target, kind, condition=_surface(a, b))
    graph.add_cube(neighbor, neighbor_kind)
    graph.add_pipe(target, neighbor)
    mesh = _export(graph)
    expected = np.array(target.as_tuple()) * 3 + 0.5
    expected[axis.value] -= sign * 0.5
    endpoints = np.concatenate(
        [
            np.array(((0.5, 0.5, 0), (0.5, 0.5, 1))) @ node.matrix[:3, :3].T + node.matrix[:3, 3]
            for node in _instances(mesh, "_CONDITION_LINK")
        ]
    )
    assert np.linalg.norm(endpoints - expected, axis=1).min() < 1e-5


def test_empty_condition_keeps_blocks_opaque() -> None:
    mesh = _export(_graph(CorrelationSurface(frozenset())))
    assert not _instances(mesh, "_CORRELATION")
    assert not _instances(mesh, "_CONDITION_LINK")
    assert mesh.effects["X_effect"].diffuse[3] == 1.0
    assert mesh.effects["Z_effect"].diffuse[3] == 1.0
    assert not any(effect.id.endswith("_translucent") for effect in mesh.effects)


@pytest.mark.parametrize("missing", ["cube", "pipe"])
def test_missing_condition_references_fail_before_output(tmp_path: Path, missing: str) -> None:
    graph = _graph()
    if missing == "cube":
        graph.remove_cube(Position3D(0, 0, 0))
    else:
        graph.remove_pipe(Position3D(0, 0, 0), Position3D(1, 0, 0))
    output = tmp_path / "invalid.dae"
    with pytest.raises(TQECError, match=r"[Cc]ondition"):
        graph.to_dae_file(output)
    assert not output.exists()
    graph.to_dae_file(output, show_conditions=False)
    assert output.exists()


@pytest.mark.parametrize("show_conditions", [False, True])
def test_html_embeds_conditional_model(tmp_path: Path, show_conditions: bool) -> None:
    output = tmp_path / "graph.html"
    viewer = _graph().view_as_html(output, show_conditions=show_conditions)
    html = str(viewer)
    encoded = re.search(r"data:text/plain;base64,([A-Za-z0-9+/=]+)", html)
    assert encoded is not None
    mesh = collada.Collada(BytesIO(base64.b64decode(encoded[1])))
    assert _instances(mesh, "ZXZ_ZXX")
    assert bool(_instances(mesh, "_CONDITION_LINK")) == show_conditions
    assert output.read_text() == html


@pytest.mark.parametrize("show_conditions", [False, True])
def test_explicit_stored_condition_uses_partial_renderer(show_conditions: bool) -> None:
    graph = _graph()
    output = BytesIO()
    write_block_graph_to_dae_file(
        graph,
        output,
        show_correlation_surface=graph.conditional_cubes[0].condition,
        show_conditions=show_conditions,
    )
    output.seek(0)
    mesh = collada.Collada(output)
    assert _surface_signatures(mesh) == _surface_signatures(_export(graph))
    assert bool(_instances(mesh, "_CONDITION_LINK")) == show_conditions
    expected_opacity = 0.3 if show_conditions else 1.0
    assert _opacities(_cube_instance(mesh, Position3D(0, 0, 0))) == {expected_opacity}


def test_explicit_complete_surface_is_additive() -> None:
    graph = _graph()
    isolated = Position3D(4, 0, 0)
    graph.add_cube(isolated, "ZXZ")
    output = BytesIO()
    write_block_graph_to_dae_file(
        graph, output, show_correlation_surface=_surface(isolated, isolated, Basis.Z)
    )
    output.seek(0)
    mesh = collada.Collada(output)
    assert _instances(mesh, "X_CORRELATION")
    assert _instances(mesh, "Z_CORRELATION")
    assert _instances(mesh, "_CONDITION_LINK")
    assert _opacities(_cube_instance(mesh, isolated)) == {1.0}


@pytest.mark.parametrize("show_conditions", [False, True])
def test_conditional_dae_reimport_fails_descriptively(
    tmp_path: Path, show_conditions: bool
) -> None:
    output = tmp_path / "conditional.dae"
    _graph().to_dae_file(output, show_conditions=show_conditions)
    with pytest.raises(TQECError, match=r"[Cc]onditional"):
        BlockGraph.from_dae_file(output)


@pytest.mark.parametrize("pipe_length", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("kind", ["ZXX_Y", "Y_ZXX"])
def test_y_branch_conditional_is_full_size_without_y_offset(pipe_length: float, kind: str) -> None:
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 1), "ZXX")
    target = Position3D(0, 0, 2)
    graph.add_cube(target, kind, condition=CorrelationSurface(frozenset()))
    graph.add_pipe(Position3D(0, 0, 1), target)
    output = BytesIO()
    write_block_graph_to_dae_file(graph, output, pipe_length)
    output.seek(0)
    mesh = collada.Collada(output)
    node = _instances(mesh, kind)[0]
    np.testing.assert_allclose(node.matrix[:3, 3], (0, 0, 2 * (1 + pipe_length)))
    vertices = np.concatenate(
        [
            primitive.vertex
            for geometry in node.children[0].node.children
            for primitive in geometry.geometry.primitives
        ]
    )
    np.testing.assert_allclose(vertices.min(axis=0), (0, 0, 0))
    np.testing.assert_allclose(vertices.max(axis=0), (1, 1, 1))
