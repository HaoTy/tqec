from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import find_correlation_surfaces
from tqec.gallery.cnot import cnot
from tqec.gallery.memory import memory
from tqec.utils.position import Position3D
from tqec.visualisation.patches.boundaries import mark_trace_boundaries
from tqec.visualisation.patches.lowering import lower_block_graph
from tqec.visualisation.patches.scene import build_patch_scene


def test_normal_preparation_and_readout_are_not_tracking_boundaries() -> None:
    scene = build_patch_scene(memory(), 1, FIXED_BULK_CONVENTION)
    assert not any(snapshot.get("boundaries") for snapshot in scene["observables"][0]["cycles"])


def test_unverified_operator_interval_marks_both_adjacent_physical_snapshots() -> None:
    graph = memory()
    computation = lower_block_graph(graph, 1)
    surface = find_correlation_surfaces(graph.to_zx_graph())[0]
    snapshots = [
        {"cycle_id": 0, "support": [[1, 1, "Z"]]},
        {"cycle_id": 2, "support": [[1, 1, "Z"]]},
    ]
    marked = mark_trace_boundaries(computation, surface, snapshots)
    assert [b["side"] for b in marked[0]["boundaries"]] == ["future"]
    assert [b["side"] for b in marked[1]["boundaries"]] == ["past"]
    assert all("could not be verified" in s["boundaries"][0]["reason"] for s in marked)
    assert not any("boundaries" in snapshot for snapshot in snapshots)


def test_empty_established_snapshot_does_not_create_tracking_loss() -> None:
    graph = memory()
    computation = lower_block_graph(graph, 1)
    surface = find_correlation_surfaces(graph.to_zx_graph())[0]
    snapshots = [
        {"cycle_id": 0, "support": [[1, 1, "Z"]]},
        {"cycle_id": 1, "support": []},
        {"cycle_id": 2, "support": [[1, 1, "Z"]]},
    ]
    assert not any(
        record.get("boundaries")
        for record in mark_trace_boundaries(computation, surface, snapshots)
    )


def test_open_cnot_marks_only_locally_available_trace_endpoints() -> None:
    scene = build_patch_scene(cnot(), 1, FIXED_BULK_CONVENTION)
    snapshots = [
        snapshot for observable in scene["observables"] for snapshot in observable["cycles"]
    ]
    marked = [snapshot for snapshot in snapshots if snapshot.get("boundaries")]
    assert marked
    assert all(snapshot["support"] for snapshot in marked)
    assert any(
        "without a physical implementation" in boundary["reason"]
        for snapshot in marked
        for boundary in snapshot["boundaries"]
    )


def test_past_boundary_does_not_skip_a_missing_final_source_round() -> None:
    graph = BlockGraph()
    a, b = Position3D(0, 0, 0), Position3D(0, 0, 1)
    graph.add_cube(a, "ZXZ")
    graph.add_cube(b, "ZXZ")
    graph.add_pipe(a, b)
    computation = lower_block_graph(graph, 1)
    surface = find_correlation_surfaces(graph.to_zx_graph())[0]
    # Another unavailable temporal connection can mask this component's final
    # round while leaving the pipe connecting a to b physically supported.
    computation.cycles = [
        cycle for cycle in computation.cycles if not (cycle.block_z == 0 and cycle.round_index == 2)
    ]
    snapshots = [
        {"cycle_id": cycle.index, "support": [[1, 1, "Z"]]} for cycle in computation.cycles
    ]
    marked = mark_trace_boundaries(computation, surface, snapshots)
    first_target = next(
        cycle.index for cycle in computation.cycles if cycle.block_z == 1 and cycle.round_index == 0
    )
    record = next(record for record in marked if record["cycle_id"] == first_target)
    assert [boundary["side"] for boundary in record["boundaries"]] == ["past"]
