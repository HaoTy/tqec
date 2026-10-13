import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BOUNDARY_CONVENTION, FIXED_BULK_CONVENTION
from tqec.compile.detectors.database import DetectorDatabase
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.gallery.cnot import cnot
from tqec.gallery.memory import memory
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Position3D
from tqec.visualisation.patches.lowering import lower_block_graph
from tqec.visualisation.patches.scene import (
    _fallback_surface,
    _SceneCoordinates,
    build_patch_scene,
)


@pytest.mark.parametrize("basis", list(Basis))
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION])
def test_memory_scene_is_complete_and_json_serializable(basis, k, convention):
    graph = memory(basis)
    before = graph.clone()
    scene = build_patch_scene(graph, k, convention)
    assert graph == before
    assert len(scene["cycles"]) == 2 * k + 1
    assert scene["coarse"] == []
    assert scene["observables"][0]["surfaces"] == []
    assert len(scene["observables"][0]["cycles"]) == 2 * k + 1
    assert all(len(c["support"]) == 2 * k + 1 for c in scene["observables"][0]["cycles"])
    assert any(q.get("reset") == basis.value for q in scene["cycles"][0]["qubits"])
    assert any(q.get("measurement") == basis.value for q in scene["cycles"][-1]["qubits"])
    json.dumps(scene, allow_nan=False)


def test_public_viewer_writes_html_without_changing_block_viewer(tmp_path: Path, monkeypatch):
    def forbid_database_access(*args, **kwargs):
        pytest.fail("The visualization must not read or write a detector database.")

    monkeypatch.setattr(DetectorDatabase, "from_file", forbid_database_access)
    monkeypatch.setattr(DetectorDatabase, "to_file", forbid_database_access)
    graph = memory()
    destination = tmp_path / "patches.html"
    viewer = graph.view_patches_as_html(write_html_filepath=destination)
    assert destination.read_text() == str(viewer) == viewer.html_str
    assert "<iframe" in viewer._repr_html_()
    assert "patch-scene" in str(viewer)
    assert "ColladaLoader" in str(graph.view_as_html())


def test_open_ports_preserve_interior_and_topological_surface_segments():
    graph = cnot()
    before = graph.clone()
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION)
    assert graph == before
    assert scene["cycles"]
    assert any(region["type"] == "port" for region in scene["coarse"])
    assert any(region["type"] == "slab" for region in scene["coarse"])
    assert all(c["global_round_index"] is None for c in scene["cycles"])
    assert any(o["cycles"] and o["surfaces"] for o in scene["observables"])
    with pytest.raises(TQECError, match="open ports"):
        compile_block_graph(graph)


def test_conditional_fallback_preserves_both_branches_and_stored_condition():
    graph = BlockGraph("Conditional readout")
    a, b, c = Position3D(0, 0, 0), Position3D(0, 0, 1), Position3D(0, 0, 2)
    condition = CorrelationSurface(frozenset({ZXEdge(ZXNode(a, Basis.Z), ZXNode(b, Basis.Z))}))
    graph.add_cube(a, "ZXZ")
    graph.add_cube(b, "ZXZ")
    graph.add_cube(c, "ZXZ_ZXX", condition=condition)
    graph.add_pipe(a, b)
    graph.add_pipe(b, c)
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION)
    conditional = next(item for item in scene["coarse"] if item.get("conditional"))
    assert conditional["conditional"] == ["ZXZ", "ZXX"]
    colors = {face["color"] for face in conditional["faces"]}
    assert {"#ffffff", "#ff7f7f", "#7396ff"} <= colors
    assert scene["cycles"]
    assert scene["observables"][0]["name"].startswith("Condition for")
    assert any("discovery" in notice for notice in scene["notices"])


def test_all_coarse_graph_and_empty_graph_still_have_viewers():
    graph = BlockGraph("Open boundary")
    a, b = Position3D(0, 0, 0), Position3D(0, 0, 1)
    graph.add_cube(a, "P", "in")
    graph.add_cube(b, "Y")
    graph.add_pipe(a, b, "XZO")
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION)
    assert not scene["cycles"]
    assert len(scene["coarse"]) == 3
    assert scene["observables"][0]["surfaces"]
    assert "<html" in str(graph.view_patches_as_html(observables=None))
    empty = build_patch_scene(BlockGraph(), 1, FIXED_BULK_CONVENTION)
    assert not empty["cycles"] and not empty["coarse"]


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "1"])
def test_invalid_distance_is_rejected(k):
    with pytest.raises(TQECError, match="positive integer"):
        memory().view_patches_as_html(k=k)


def test_explicit_surface_references_are_validated():
    graph = memory()
    missing = Position3D(8, 0, 0)
    surface = CorrelationSurface(
        frozenset({ZXEdge(ZXNode(missing, Basis.Z), ZXNode(missing, Basis.Z))})
    )
    with pytest.raises(TQECError, match="missing cube"):
        graph.view_patches_as_html(observables=[surface])
    with pytest.raises(TQECError, match="observables must"):
        graph.view_patches_as_html(observables=cast(Any, "wrong"))


def test_missing_pipe_is_rejected_but_unrelated_regions_are_kept():
    graph = BlockGraph()
    a, b = Position3D(0, 0, 0), Position3D(3, 0, 0)
    graph.add_cube(a, "ZXZ")
    graph.add_cube(b, "ZXZ")
    surface = CorrelationSurface(frozenset({ZXEdge(ZXNode(a, Basis.Z), ZXNode(b, Basis.Z))}))
    with pytest.raises(TQECError, match="missing pipe"):
        graph.view_patches_as_html(observables=[surface])
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION, None)
    assert len(scene["cycles"]) == 6


def test_no_observables_and_negative_coordinates():
    graph = cnot().shift_by(dx=-3, dy=-2, dz=-4)
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION, None)
    assert scene["observables"] == []
    assert min(c["block_z"] for c in scene["cycles"]) < 0
    assert min(q["x"] for c in scene["cycles"] for q in c["qubits"]) < 0
    assert min(c["time"] for c in scene["cycles"]) >= 0
    for item in scene["coarse"]:
        assert np.isfinite(item["center"]).all()
        assert all(v > 0 for v in item["size"])


def test_known_temporal_hadamard_has_separate_display_round():
    graph = BlockGraph()
    a, b = Position3D(0, 0, -1), Position3D(0, 0, 0)
    graph.add_cube(a, "ZXZ")
    graph.add_cube(b, "XZX")
    graph.add_pipe(a, b, "ZXOH")
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION, None)
    assert len(scene["cycles"]) == 7
    assert [c["time"] for c in scene["cycles"]] == [i + 0.5 for i in range(7)]
    assert sum("realignment" in c["label"] for c in scene["cycles"]) == 1


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_port_marker_faces_its_incident_pipe(axis: int, sign: int) -> None:
    graph = BlockGraph()
    coordinates = [0, 0, 0]
    coordinates[axis] = sign
    port, neighbor = Position3D(0, 0, 0), Position3D(*coordinates)
    graph.add_cube(port, "P", "open")
    graph.add_cube(neighbor, "XZZ" if axis == 1 else "ZXZ")
    graph.add_pipe(port, neighbor)
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION, None)
    marker = next(item for item in scene["coarse"] if item["type"] == "port")
    outline = next(item for item in scene["outlines"] if item["label"].startswith("open at"))
    assert marker["size"][axis] == 0.1
    assert marker["center"][axis] == outline["center"][axis] + sign * outline["size"][axis] / 2
    assert all(
        marker["size"][other] == outline["size"][other] for other in range(3) if other != axis
    )


def test_missing_logical_hadamard_transition_gets_topological_surface() -> None:
    graph = BlockGraph()
    u, v = Position3D(0, 0, 0), Position3D(0, 0, 1)
    graph.add_cube(u, "XZZ")
    graph.add_cube(v, "ZXX")
    graph.add_pipe(u, v)
    computation = lower_block_graph(graph, 1)
    surface = graph.find_correlation_surfaces()[0]
    coordinates = _SceneCoordinates(3, 0, frozenset({0}))
    ordinary_cycle_ids = {cycle.index for cycle in computation.cycles if not cycle.is_transition}
    fallback = _fallback_surface(computation, surface, ordinary_cycle_ids, coordinates)
    assert fallback
    assert {vertex[2] for piece in fallback for vertex in piece["vertices"]} == {3, 3.5, 4}
    assert not _fallback_surface(
        computation, surface, {cycle.index for cycle in computation.cycles}, coordinates
    )


@pytest.mark.parametrize("is_preparation", [False, True])
def test_y_singleton_tiles_are_inside_actual_half_cube(is_preparation: bool) -> None:
    graph = BlockGraph()
    y_pos = Position3D(0, 0, 0 if is_preparation else 1)
    neighbor = Position3D(0, 0, 1 if is_preparation else 0)
    graph.add_cube(y_pos, "Y")
    graph.add_cube(neighbor, "ZXZ")
    graph.add_pipe(y_pos, neighbor)
    surface = CorrelationSurface(
        frozenset(ZXEdge(ZXNode(y_pos, basis), ZXNode(y_pos, basis)) for basis in Basis)
    )
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION, [surface])
    half_cube = next(item for item in scene["coarse"] if item["kind"] == "Y")
    lower, upper = (half_cube["center"][2] + sign * half_cube["size"][2] / 2 for sign in [-1, 1])
    tiles = scene["observables"][0]["surfaces"]
    assert len(tiles) == 2
    assert all(lower < vertex[2] < upper for tile in tiles for vertex in tile["vertices"])
    assert tiles[0]["vertices"][0][2] != tiles[1]["vertices"][0][2]


def test_automatic_discovery_on_disconnected_memories():
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), "ZXZ")
    graph.add_cube(Position3D(3, 0, 0), "ZXZ")
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION)
    assert len(scene["cycles"]) == 6
    assert len(scene["observables"]) == 2
    assert all(len(overlay["cycles"]) == 3 for overlay in scene["observables"])


def test_no_leaf_discovery_limitation_retains_scene_and_explicit_surfaces():
    graph = BlockGraph()
    positions = [
        Position3D(0, 0, 0),
        Position3D(1, 0, 0),
        Position3D(1, 1, 0),
        Position3D(0, 1, 0),
    ]
    for position in positions:
        graph.add_cube(position, "ZZX")
    edges = list(zip(positions, positions[1:] + positions[:1]))
    for a, b in edges:
        graph.add_pipe(a, b)
    scene = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION)
    assert len(scene["cycles"]) == 3
    assert not scene["observables"]
    assert any("without leaf nodes" in notice for notice in scene["notices"])
    surface = CorrelationSurface(
        frozenset(ZXEdge.sorted(ZXNode(a, Basis.X), ZXNode(b, Basis.X)) for a, b in edges)
    )
    explicit = build_patch_scene(graph, 1, FIXED_BULK_CONVENTION, [surface])
    assert len(explicit["observables"]) == 1
