"""Physical compilation and honest boundaries around unsupported operations."""

from collections import defaultdict
from dataclasses import replace
from itertools import pairwise
from typing import Any

import pytest
import stim

from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.compile.blocks.block import Block
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import (
    FIXED_BOUNDARY_CONVENTION,
    FIXED_BULK_CONVENTION,
    Convention,
    ConventionTriplet,
    fixed_bulk_convention,
)
from tqec.compile.detectors.database import DetectorDatabase
from tqec.compile.specs.base import CubeSpec, PipeSpec
from tqec.compile.specs.library.generators.schedules import DIAGONAL_SCHEDULE_FAMILY
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ZXCube
from tqec.gallery.cnot import cnot
from tqec.gallery.move_rotation import move_rotation
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Position3D
from tqec.utils.scale import LinearFunction
from tqec.visualisation.patches.lowering import lower_block_graph

DIAGONAL_CONVENTION = fixed_bulk_convention(DIAGONAL_SCHEDULE_FAMILY)


def _operation_signature(circuit: ScheduledCircuit) -> list[tuple[Any, ...]]:
    """Compare physical operations independently of qubit numbering and batching."""
    result: list[tuple[Any, ...]] = []
    for schedule, moment in circuit.scheduled_moments:
        for instruction in moment.circuit:
            assert isinstance(instruction, stim.CircuitInstruction)
            for group in instruction.target_groups():
                coordinates = tuple(
                    (circuit.qubit_map.i2q[t.value].x, circuit.qubit_map.i2q[t.value].y)
                    if t.is_qubit_target
                    else str(t)
                    for t in group
                )
                result.append(
                    (schedule, instruction.name, tuple(instruction.gate_args_copy()), coordinates)
                )
    return sorted(result)


def _assert_matches_strict(graph: BlockGraph, k: int, convention: Convention) -> None:
    actual = lower_block_graph(graph, k, convention)
    assert actual.fully_supported
    expected = list(
        compile_block_graph(graph, convention, observables=None).to_layer_tree().iter_cycles(k)
    )
    parallel: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
    min_z = min(cube.position.z for cube in graph.cubes)
    for cycle in actual.cycles:
        parallel[cycle.block_z - min_z, cycle.round_index].extend(
            _operation_signature(cycle.circuit)
        )
    assert len(parallel) == len(expected)
    for cycle in expected:
        assert sorted(parallel[cycle.block_z, cycle.round_index]) == _operation_signature(
            cycle.circuit
        )


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("kind", ["ZXZ", "ZXX"])
@pytest.mark.parametrize("convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION])
def test_memory_rounds(k: int, kind: str, convention: Convention) -> None:
    graph = BlockGraph()
    graph.add_cube(Position3D(-2, -1, -3), kind)
    computation = lower_block_graph(graph, k, convention)
    assert computation.fully_supported
    assert len(computation.cycles) == 2 * k + 1
    assert all(c.block_z == -3 for c in computation.cycles)
    assert [c.round_index for c in computation.cycles] == list(range(2 * k + 1))
    strict = compile_block_graph(graph, convention, observables=None).to_layer_tree()
    expected = list(strict.iter_cycles(k))
    for actual, reference in zip(computation.cycles, expected):
        assert actual.circuit.get_circuit() == reference.circuit.get_circuit()
        assert actual.measurement_start == reference.measurement_start
        assert actual.measurement_end == reference.measurement_end
        assert actual.tick_start == reference.tick_start
        assert actual.tick_end == reference.tick_end
        assert len([q for q in actual.circuit.qubits if q.x % 2 and q.y % 2]) == (2 * k + 1) ** 2
    if k == 2:
        assert computation.cycles[1].circuit is computation.cycles[2].circuit


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize(("dx", "dy", "kind"), [(1, 0, "ZXZ"), (0, 1, "XZZ")])
def test_surgery_bridge(k: int, dx: int, dy: int, kind: str) -> None:
    graph = BlockGraph()
    u, v = Position3D(0, 0, 0), Position3D(dx, dy, 0)
    graph.add_cube(u, kind)
    graph.add_cube(v, kind)
    graph.add_pipe(u, v)
    computation = lower_block_graph(graph, k)
    d = 2 * k + 1
    assert len(computation.cycles) == d
    for cycle in computation.cycles:
        data = [q for q in cycle.circuit.qubits if q.x % 2 and q.y % 2]
        assert len(data) == 2 * d * d + d
        bridge = [q for q in data if (q.x if dx else q.y) == 2 * d - 1]
        assert len(bridge) == d
        assert cycle.component == (u, v)


@pytest.mark.parametrize("convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION])
@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
def test_cnot(convention: Convention, basis: Basis) -> None:
    graph = cnot(basis)
    initial = graph.clone()
    computation = lower_block_graph(graph, 2, convention)
    assert computation.fully_supported
    assert computation.cycles
    assert graph == initial
    assert all(c.global_round_index is not None for c in computation.cycles)


def test_y_boundary_has_no_false_cap_or_temporal_cascade() -> None:
    graph = BlockGraph()
    positions = [Position3D(0, 0, i - 2) for i in range(4)]
    for pos in positions[:-1]:
        graph.add_cube(pos, "ZXZ")
    graph.add_cube(positions[-1], "Y")
    for u, v in pairwise(positions):
        graph.add_pipe(u, v)
    computation = lower_block_graph(graph, 1)
    assert len(computation.cycles) == 8
    assert [c.round_index for c in computation.cycles if c.block_z == 0] == [0, 1]
    assert [c.round_index for c in computation.cycles if c.block_z == -1] == [0, 1, 2]
    assert all(c.global_round_index is None for c in computation.cycles)
    assert any(
        r.cubes == (positions[2],) and r.round_index == 2 for r in computation.coarse_regions
    )
    for cycle in computation.cycles:
        if cycle.block_z == 0:
            # Both visible rounds are memory rounds: no fabricated readout.
            assert cycle.circuit.num_measurements == 8


def test_open_spatial_port_falls_back_whole_component() -> None:
    graph = BlockGraph()
    u, v = Position3D(0, 0, 0), Position3D(1, 0, 0)
    graph.add_cube(u, "ZXZ")
    graph.add_cube(v, "PORT", "out")
    graph.add_pipe(u, v, "OXZ")
    computation = lower_block_graph(graph, 1)
    assert not computation.cycles
    assert computation.coarse_regions[0].cubes == (u, v)
    assert computation.coarse_regions[0].round_index is None
    assert "PORT" in computation.coarse_regions[0].reason


def test_conditional_boundary_mask_spreads_across_spatial_component() -> None:
    graph = BlockGraph()
    a0, a1, b1, cond = [
        Position3D(0, 0, 0),
        Position3D(0, 0, 1),
        Position3D(1, 0, 1),
        Position3D(1, 0, 2),
    ]
    condition = CorrelationSurface(frozenset({ZXEdge(ZXNode(a1, Basis.X), ZXNode(b1, Basis.X))}))
    for pos in [a0, a1, b1]:
        graph.add_cube(pos, "ZXZ")
    graph.add_cube(cond, "ZXZ_ZXX", condition=condition)
    for u, v in [(a0, a1), (a1, b1), (b1, cond)]:
        graph.add_pipe(u, v)
    computation = lower_block_graph(graph, 1)
    assert len(computation.cycles) == 5
    assert any(r.cubes == (a1, b1) and r.round_index == 2 for r in computation.coarse_regions)
    assert graph[cond].condition == condition


def test_unsupported_spatial_pipe_between_known_slices() -> None:
    graph = BlockGraph()
    for x, kind in [(0, "ZXZ"), (1, "XZX")]:
        for z in range(3):
            graph.add_cube(Position3D(x, 0, z), kind)
        for z in range(2):
            graph.add_pipe(Position3D(x, 0, z), Position3D(x, 0, z + 1))
    graph.add_pipe(Position3D(0, 0, 1), Position3D(1, 0, 1))
    computation = lower_block_graph(graph, 1)
    assert len(computation.cycles) == 8
    assert all(c.block_z != 1 for c in computation.cycles)
    assert {c.round_index for c in computation.cycles if c.block_z == 0} == {0, 1}
    assert {c.round_index for c in computation.cycles if c.block_z == 2} == {1, 2}
    assert any(
        r.cubes == (Position3D(0, 0, 1), Position3D(1, 0, 1)) for r in computation.coarse_regions
    )


def test_temporal_hadamard_has_separate_aligned_transition_rounds() -> None:
    graph = BlockGraph()
    for x, kinds in [(0, ("XZZ", "ZXX")), (2, ("XZZ", "XZZ"))]:
        for z, kind in enumerate(kinds):
            graph.add_cube(Position3D(x, 0, z), kind)
        graph.add_pipe(Position3D(x, 0, 0), Position3D(x, 0, 1))
    computation = lower_block_graph(graph, 1)
    transitions = [c for c in computation.cycles if c.is_transition]
    assert len(computation.cycles) == 14
    assert len(transitions) == 2
    assert {(c.block_z, c.round_index, c.global_round_index) for c in transitions} == {(0, 3, 3)}
    strict_cycles = list(
        compile_block_graph(graph, observables=None).to_layer_tree().iter_cycles(1)
    )
    assert len(strict_cycles) == 7
    assert [c.index for c in strict_cycles if c.is_transition] == [3]
    for transition in transitions:
        preceding = next(
            c
            for c in computation.cycles
            if c.component == transition.component and c.round_index == 2
        )
        assert transition.fragment_id == preceding.fragment_id
        assert transition.tick_start == preceding.tick_end
        assert transition.measurement_start == preceding.measurement_end
    later = [c for c in computation.cycles if c.block_z == 1 and c.round_index == 0]
    assert all(c.tick_start == c.measurement_start == 0 for c in later)
    assert not {c.fragment_id for c in later} & {c.fragment_id for c in transitions}


def test_unsupported_temporal_pipe_masks_only_touching_rounds() -> None:
    def pipe_builder(spec: PipeSpec, block_temporal_height: LinearFunction) -> Block:
        if spec.pipe_kind.has_hadamard:
            raise NotImplementedError("Temporal Hadamards are unavailable.")
        return FIXED_BULK_CONVENTION.triplet.pipe_builder(spec, block_temporal_height)

    convention = Convention(
        "without_hadamards", replace(FIXED_BULK_CONVENTION.triplet, pipe_builder=pipe_builder)
    )
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), "XZZ")
    graph.add_cube(Position3D(0, 0, 1), "ZXX")
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    computation = lower_block_graph(graph, 1, convention)
    assert [(c.block_z, c.round_index) for c in computation.cycles] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
    ]
    assert len(computation.coarse_regions) == 3


def test_full_topology_specs_survive_fallback() -> None:
    observed: list[CubeSpec] = []

    def cube_builder(spec: CubeSpec, block_temporal_height: LinearFunction) -> Block:
        observed.append(spec)
        return FIXED_BULK_CONVENTION.triplet.cube_builder(spec, block_temporal_height)

    convention = Convention(
        "recording", replace(FIXED_BULK_CONVENTION.triplet, cube_builder=cube_builder)
    )
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), "ZXZ")
    graph.add_cube(Position3D(0, 0, 1), "Y")
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    lower_block_graph(graph, 1, convention)
    assert len(observed) == 1
    assert observed[0].has_top_temporal_pipe


def test_errors_are_not_hidden_and_no_database_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unexpected database access")

    monkeypatch.setattr(DetectorDatabase, "from_file", fail)
    monkeypatch.setattr(DetectorDatabase, "to_file", fail)
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), "ZXZ")
    assert lower_block_graph(graph, 1).fully_supported

    def broken_builder(spec: CubeSpec, block_temporal_height: LinearFunction) -> Block:
        raise ValueError("Implementation bug")

    triplet = ConventionTriplet(
        broken_builder,
        FIXED_BULK_CONVENTION.triplet.pipe_builder,
        FIXED_BULK_CONVENTION.triplet.observable_builder,
    )
    with pytest.raises(ValueError, match="Implementation bug"):
        lower_block_graph(graph, 1, Convention("broken", triplet))
    graph.add_cube(Position3D(2, 0, 0), "PORT", "unconnected")
    with pytest.raises(TQECError, match="exactly one"):
        lower_block_graph(graph, 1)


@pytest.mark.parametrize("k", [True, 0, -1, 1.5])
def test_invalid_scaling(k: Any) -> None:
    with pytest.raises(TQECError, match="positive integer"):
        lower_block_graph(BlockGraph(), k)


def test_empty_graph() -> None:
    computation = lower_block_graph(BlockGraph(), 1)
    assert not computation.cycles
    assert computation.notices == ["The graph is empty."]


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
@pytest.mark.parametrize("factory", [cnot, move_rotation])
@pytest.mark.parametrize(
    "convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION, DIAGONAL_CONVENTION]
)
def test_complete_physical_rounds_match_strict_compilation(
    k: int, basis: Basis, factory: Any, convention: Convention
) -> None:
    graph = factory(basis).shift_by(dx=-3, dy=-2, dz=-4)
    _assert_matches_strict(graph, k, convention)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize(
    "convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION, DIAGONAL_CONVENTION]
)
@pytest.mark.parametrize(
    ("name", "u_kind", "v_kind", "delta"),
    [
        ("temporal", "XZZ", "ZXX", (0, 0, 1)),
        ("regular-x", "ZXZ", "XZX", (1, 0, 0)),
        ("regular-y", "XZZ", "ZXX", (0, 1, 0)),
        ("junction-x", "ZZX", "XXZ", (1, 0, 0)),
        ("junction-y", "ZZX", "XXZ", (0, 1, 0)),
    ],
)
def test_hadamard_capabilities_and_round_equivalence(
    k: int,
    convention: Convention,
    name: str,
    u_kind: str,
    v_kind: str,
    delta: tuple[int, int, int],
) -> None:
    graph = BlockGraph()
    u, v = Position3D(-2, -3, -4), Position3D(-2 + delta[0], -3 + delta[1], -4 + delta[2])
    graph.add_cube(u, u_kind)
    graph.add_cube(v, v_kind)
    graph.add_pipe(u, v)
    supported = (
        name == "temporal"
        or (convention is FIXED_BOUNDARY_CONVENTION and name.startswith("regular"))
        or (convention is FIXED_BULK_CONVENTION and name == "junction-y")
    )
    if supported:
        _assert_matches_strict(graph, k, convention)
    else:
        computation = lower_block_graph(graph, k, convention)
        assert not computation.cycles
        assert computation.coarse_regions
        if convention is DIAGONAL_CONVENTION:
            assert "diagonal" in computation.coarse_regions[0].reason


def test_parallel_custom_schedules_align_measurements() -> None:
    def cube_builder(spec: CubeSpec, block_temporal_height: LinearFunction) -> Block:
        convention = FIXED_BULK_CONVENTION if spec.kind == ZXCube.ZXZ else DIAGONAL_CONVENTION
        return convention.triplet.cube_builder(spec, block_temporal_height)

    convention = Convention(
        "mixed_schedules", replace(FIXED_BULK_CONVENTION.triplet, cube_builder=cube_builder)
    )
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), "ZXZ")
    graph.add_cube(Position3D(2, 0, 0), "ZXX")
    _assert_matches_strict(graph, 2, convention)
    computation = lower_block_graph(graph, 2, convention)
    assert all(c.circuit.schedule.max_schedule == 8 for c in computation.cycles)
    assert all(c.tick_end - c.tick_start == 9 for c in computation.cycles)
