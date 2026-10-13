"""Physical checks and logical overlays are backed by actual signed circuit flows."""

from dataclasses import replace
from itertools import pairwise

import pytest
import stim

from tqec.circuit.qubit import GridQubit
from tqec.compile.convention import (
    FIXED_BOUNDARY_CONVENTION,
    FIXED_BULK_CONVENTION,
    Convention,
    fixed_bulk_convention,
)
from tqec.compile.observables.abstract_observable import CubeWithArms
from tqec.compile.specs.library.generators.schedules import DIAGONAL_SCHEDULE_FAMILY
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.gallery.cnot import cnot
from tqec.gallery.memory import memory
from tqec.utils.enums import Basis
from tqec.utils.position import PlaquetteShape2D, Position3D
from tqec.visualisation.patches.analysis import (
    _build_snapshot_circuit,
    _pauli_from_support,
    _split_snapshot,
    analyse_computation,
    explain_transition,
    extract_checks,
)
from tqec.visualisation.patches.lowering import lower_block_graph


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
@pytest.mark.parametrize("convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION])
def test_memory_checks_and_all_logical_snapshots(
    k: int, basis: Basis, convention: Convention
) -> None:
    graph = memory(basis).shift_by(dx=-2, dy=-1, dz=-3)
    original = graph.clone()
    computation = lower_block_graph(graph, k, convention)
    result = analyse_computation(computation, graph.find_correlation_surfaces())
    d = 2 * k + 1
    assert len(result.cycles) == d
    for analysis in result.cycles.values():
        assert len(analysis.roles.data) == d * d
        assert len(analysis.checks) == d * d - 1
        assert not analysis.added
        assert not analysis.removed_checks
        for check in analysis.checks:
            assert check.support == check.output_support
            assert len(check.support) in (2, 4)
            assert check.input_flow is not None and check.output_flow is not None
            assert analysis.diagnostic_circuit.has_flow(check.input_flow)
            assert analysis.diagnostic_circuit.has_flow(check.output_flow)
    snapshots = result.observables[0]["cycles"]
    assert len(snapshots) == d
    # The last visible round is before destructive data readout, not empty.
    assert all(len(snapshot["support"]) == d for snapshot in snapshots)
    assert len({tuple(snapshot["support"]) for snapshot in snapshots}) == 1
    assert all({p for _, _, p in snapshot["support"]} == {basis.value} for snapshot in snapshots)
    assert all(snapshot["validated"] and not snapshot["factors"] for snapshot in snapshots)
    assert graph == original


def test_extended_check_is_joint_parity_not_independent_halves() -> None:
    # A GHZ ancilla measures X2 X3. Neither individual X-ancilla readout
    # measures a data-only check; only their parity does.
    circuit = stim.Circuit("RX 0\nR 1\nCX 0 1 0 2 1 3\nMX 0 1")
    coordinates = {i: GridQubit(i, 0) for i in range(4)}
    checks, diagnostic, mapping = extract_checks(
        circuit, coordinates, frozenset({coordinates[2], coordinates[3]})
    )
    assert mapping == (0, 1)
    assert len(checks) == 1
    check = checks[0]
    assert check.measurements == (0, 1)
    assert check.support == check.output_support == ((2, 0, "X"), (3, 0, "X"))
    assert check.input_flow is not None and check.output_flow is not None
    assert diagnostic.has_flow(check.input_flow)
    assert diagnostic.has_flow(check.output_flow)


def test_mixed_pauli_and_negative_check_phase() -> None:
    circuit = stim.Circuit("MPP !X0*Y1*Z2")
    coordinates = {i: GridQubit(i, 0) for i in range(3)}
    checks, _, _ = extract_checks(circuit, coordinates, frozenset(coordinates.values()))
    assert len(checks) == 1
    assert checks[0].support == ((0, 0, "X"), (1, 0, "Y"), (2, 0, "Z"))
    assert checks[0].sign == checks[0].output_sign == "-"


def test_snapshot_moves_only_disjoint_final_data_measurements_and_remaps() -> None:
    circuit = stim.Circuit("R 0\nTICK\nCX 1 0\nTICK\nM 1 0")
    before, after, mapping = _split_snapshot(circuit, {1})
    assert before == stim.Circuit("R 0\nTICK\nCX 1 0\nTICK\nM 0")
    assert after == stim.Circuit("M 1")
    assert mapping == {1: 0, 0: 1}


@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
@pytest.mark.parametrize("convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION])
def test_cnot_changes_and_validated_factors(basis: Basis, convention: Convention) -> None:
    graph = cnot(basis)
    computation = lower_block_graph(graph, 1, convention)
    result = analyse_computation(computation, graph.find_correlation_surfaces())
    assert any(a.added for a in result.cycles.values())
    assert any(a.removed_checks for a in result.cycles.values())
    factors = [
        factor for obs in result.observables for c in obs["cycles"] for factor in c["factors"]
    ]
    assert factors
    assert any(factor["label"] == "Measured stabilizer" for factor in factors)
    for observable in result.observables:
        assert len(observable["cycles"]) == len(computation.cycles)
        assert all(c["validated"] for c in observable["cycles"])
        assert not any("unavailable" in c["description"] for c in observable["cycles"])


def test_logical_deformation_is_actual_stabilizer_product() -> None:
    # X0 -> X0 X1 X2 differs by the measured X1 X2 stabilizer.
    circuit = stim.Circuit("RX 3\nCX 3 1 3 2\nMX 3")
    coordinates = {i: GridQubit(i, 0) for i in range(4)}
    checks, _, _ = extract_checks(circuit, coordinates, frozenset(coordinates[i] for i in range(3)))
    previous, current = stim.PauliString("X___"), stim.PauliString("XXX_")
    explanation = explain_transition(
        circuit, previous, current, [0], coordinates, [(checks[0], [0])]
    )
    assert explanation is not None
    assert explanation["kind"] == "stabilizer_product"
    product = previous.copy()
    parity: set[int] = set()
    for factor in explanation["factors"]:
        term = _pauli_from_support(factor["support"], coordinates)
        if factor["sign"] == "-":
            term = -term
        product *= term
        parity.symmetric_difference_update(factor["measurements"])
    assert product == current
    assert parity == {0}
    assert explanation["validated"]
    assert explain_transition(circuit, previous, current, [], coordinates, []) is None


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION])
def test_cnot_deformations_relate_the_displayed_representatives(k, convention) -> None:
    graph = cnot(Basis.Z)
    computation = lower_block_graph(graph, k, convention)
    analysis = analyse_computation(computation, graph.find_correlation_surfaces())
    trace = {record["cycle_id"]: record for record in analysis.observables[1]["cycles"]}
    circuit, snapshots, coordinates, _ = _build_snapshot_circuit(computation, analysis.cycles, [])
    d = 2 * k + 1

    def at(z, round_index):
        return [s for s in snapshots if (s.cycle.block_z, s.cycle.round_index) == (z, round_index)]

    def logical(group):
        support = tuple(term for s in group for term in trace[s.cycle.index]["support"])
        return _pauli_from_support(support, coordinates)

    # The first merge contributes exactly the Z checks tiling the strip between
    # the two displayed strings, including the weight-two boundary checks.
    first = at(1, 0)
    factors = [f for s in first for f in trace[s.cycle.index]["factors"]]
    product = stim.PauliString(len(coordinates))
    for factor in factors:
        pauli = _pauli_from_support(factor["support"], coordinates)
        if factor["sign"] == "-":
            pauli = -pauli
        product *= pauli
    assert product == logical(at(0, d - 1)) * logical(first)
    if convention == FIXED_BULK_CONVENTION:
        strip = {
            check.output_support
            for s in first
            for check in analysis.cycles[s.cycle.index].checks
            if check.output_support
            and all(
                basis == "Z" and 1 <= x <= 2 * d - 1 and d <= y <= 3 * d + 2
                for x, y, basis in check.output_support
            )
        }
        assert {tuple(f["support"]) for f in factors} == strip
        assert len(factors) == (d + 1) ** 2 // 2

    # At the second merge the actual string only extends through prepared
    # bridge data. It has direct zero-parity transport; no check product is
    # needed to undo an arbitrary intermediate row choice.
    before, after = at(1, d - 1), at(2, 0)
    segment = circuit[before[0].instruction : after[0].instruction]
    previous, current = logical(before), logical(after)
    assert segment.has_flow(stim.Flow(input=previous, output=current))
    assert not segment.has_flow(stim.Flow(input=previous, output=previous))
    for snapshot in after:
        record = trace[snapshot.cycle.index]
        assert record["kind"] == "circuit_transport"
        assert not record["factors"]
        assert not record["transport_measurements"]
        assert _pauli_from_support(record["transport_support"], coordinates) == current


def test_direct_transport_separates_a_measurement_only_relation() -> None:
    # H transports Z0 to X0. The known +Z1 readout adds an independent parity,
    # which must not be presented as a geometric deformation of the logical.
    circuit = stim.Circuit("R 1\nH 0\nM 1")
    coordinates = {0: GridQubit(0, 0), 1: GridQubit(1, 0)}
    result = explain_transition(
        circuit, stim.PauliString("Z_"), stim.PauliString("X_"), [0], coordinates, []
    )
    assert result is not None
    assert result["kind"] == "circuit_transport"
    assert result["factors"] == []
    assert result["transport_measurements"] == []
    assert result["measurement_relation"] == [0]
    assert "measurement-only relation" in result["description"]


@pytest.mark.parametrize("convention", [FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION])
def test_temporal_hadamard_is_circuit_transport(convention: Convention) -> None:
    graph = BlockGraph()
    a = graph.add_cube(Position3D(0, 0, 0), "XZZ")
    b = graph.add_cube(Position3D(0, 0, 1), "ZXX")
    graph.add_pipe(a, b)
    computation = lower_block_graph(graph, 1, convention)
    analysis = analyse_computation(computation, graph.find_correlation_surfaces())
    snapshots = analysis.observables[0]["cycles"]
    assert {p for _, _, p in snapshots[0]["support"]} == {"Z"}
    assert {p for _, _, p in snapshots[-1]["support"]} == {"X"}
    assert any(c.get("kind") == "circuit_transport" for c in snapshots)
    assert all(c["validated"] for c in snapshots)


def test_local_surface_continues_on_each_side_of_coarse_region() -> None:
    # Unsupported spatial Hadamard makes the middle slice coarse while each
    # end retains two independently verified physical rounds.
    graph = BlockGraph()
    surfaces = []
    for x, kind, basis in [(0, "ZXZ", Basis.Z), (1, "XZX", Basis.X)]:
        positions = [Position3D(x, 0, z) for z in range(3)]
        for pos in positions:
            graph.add_cube(pos, kind)
        for u, v in pairwise(positions):
            graph.add_pipe(u, v)
        surfaces.append(
            CorrelationSurface(
                frozenset(
                    ZXEdge(ZXNode(u, basis), ZXNode(v, basis)) for u, v in pairwise(positions)
                )
            )
        )
    graph.add_pipe(Position3D(0, 0, 1), Position3D(1, 0, 1))
    computation = lower_block_graph(graph, 1)
    analysis = analyse_computation(computation, surfaces)
    assert not computation.fully_supported
    for observable in analysis.observables:
        records = observable["cycles"]
        assert len(records) == 4
        assert all(c["open_boundary"] and c["validated"] for c in records)
        assert all(len(c["support"]) == 3 for c in records)
        # First supported round on either side has no cross-gap relation.
        for block_z in (0, 2):
            first = next(
                c
                for c in computation.cycles
                if c.block_z == block_z and c.index in {r["cycle_id"] for r in records}
            )
            record = next(r for r in records if r["cycle_id"] == first.index)
            assert "kind" not in record
            assert not analysis.cycles[first.index].added
            assert not analysis.cycles[first.index].removed_checks


def test_disconnected_components_share_snapshot_cuts() -> None:
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), "ZXZ")
    graph.add_cube(Position3D(3, 0, 0), "ZXZ")
    computation = lower_block_graph(graph, 1)
    analysis = analyse_computation(computation)
    _, snapshots, _, _ = _build_snapshot_circuit(computation, analysis.cycles, [])
    assert len(snapshots) == 6
    assert [s.tick for s in snapshots[::2]] == [s.tick for s in snapshots[1::2]]
    assert [s.instruction for s in snapshots[::2]] == [s.instruction for s in snapshots[1::2]]


def test_partial_surface_only_falls_back_its_own_overlay() -> None:
    graph = BlockGraph()
    positions = [Position3D(0, 0, z) for z in range(3)]
    for p in positions:
        graph.add_cube(p, "ZXZ")
    for a, b in pairwise(positions):
        graph.add_pipe(a, b)
    partial = CorrelationSurface(
        frozenset({ZXEdge(ZXNode(positions[0], Basis.Z), ZXNode(positions[1], Basis.Z))})
    )
    computation = lower_block_graph(graph, 1)
    analysis = analyse_computation(computation, [*graph.find_correlation_surfaces(), partial])
    assert computation.fully_supported
    assert len(analysis.observables[0]["cycles"]) == 9
    assert not any(c.get("open_boundary") for c in analysis.observables[0]["cycles"])
    assert len(analysis.observables[1]["cycles"]) == 6
    assert all(c["open_boundary"] for c in analysis.observables[1]["cycles"])
    assert analysis.observables[1]["notices"]


def test_unconnected_temporally_adjacent_cubes_have_no_stabilizer_differences() -> None:
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), "ZXZ")
    graph.add_cube(Position3D(0, 0, 1), "XZZ")
    computation = lower_block_graph(graph, 1)
    analysis = analyse_computation(computation)
    assert all(not a.added and not a.removed_checks for a in analysis.cycles.values())


def test_signed_deformation_retains_negative_measurement_phase() -> None:
    circuit = stim.Circuit("MPP !X1*X2")
    coordinates = {i: GridQubit(i, 0) for i in range(3)}
    checks, _, _ = extract_checks(circuit, coordinates, frozenset(coordinates.values()))
    explanation = explain_transition(
        circuit,
        stim.PauliString("X__"),
        stim.PauliString("XXX"),
        [0],
        coordinates,
        [(checks[0], [0])],
    )
    assert explanation is not None
    assert explanation["sign"] == "-"
    assert explanation["factors"][0]["sign"] == "-"
    assert explanation["validated"]


@pytest.mark.parametrize(
    "convention",
    [
        fixed_bulk_convention(DIAGONAL_SCHEDULE_FAMILY),
        FIXED_BOUNDARY_CONVENTION,
    ],
)
def test_junction_schedules_use_complete_checks(convention: Convention) -> None:
    graph = BlockGraph()
    positions = [Position3D(0, 0, 0), Position3D(0, 1, 0), Position3D(1, 1, 0)]
    for position, kind in zip(positions, ["ZXX", "ZZX", "XZX"]):
        graph.add_cube(position, kind)
    for a, b in pairwise(positions):
        graph.add_pipe(a, b)
    computation = lower_block_graph(graph, 2, convention)
    analysis = analyse_computation(computation, graph.find_correlation_surfaces())
    for cycle in analysis.cycles.values():
        assert not cycle.roles.data & cycle.roles.syndrome
        assert not cycle.added
        assert not cycle.removed_checks
        for check in cycle.checks:
            assert check.input_flow is not None and check.output_flow is not None
            assert cycle.diagnostic_circuit.has_flow(check.input_flow)
            assert cycle.diagnostic_circuit.has_flow(check.output_flow)
        if convention.name == "fixed_boundary":
            # Check crosses the missing bridge-data row; individual halves
            # never appear as separate checks.
            assert any(
                max(y for _, y, _ in check.support) - min(y for _, y, _ in check.support) == 4
                for check in cycle.checks
            )


@pytest.mark.parametrize("kind", ["ZZX", "XXZ"])
def test_stability_surface_shows_its_initial_measured_checks(kind: str) -> None:
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), kind)
    analysis = analyse_computation(lower_block_graph(graph, 1), graph.find_correlation_surfaces())
    snapshots = analysis.observables[0]["cycles"]
    assert all(not snapshot["support"] for snapshot in snapshots)
    assert snapshots[0]["kind"] == "measurement_boundary"
    assert snapshots[0]["factors"]
    assert snapshots[0]["validated"]
    assert not any(snapshot["factors"] for snapshot in snapshots[1:])


def test_partial_single_cube_basis_is_not_silently_replaced() -> None:
    graph = memory(Basis.Z)
    node = ZXNode(Position3D(0, 0, 0), Basis.X)
    partial = CorrelationSurface(frozenset({ZXEdge(node, node)}))
    analysis = analyse_computation(lower_block_graph(graph, 1), [partial])
    assert all(
        {p for _, _, p in snapshot["support"]} == {"X"}
        for snapshot in analysis.observables[0]["cycles"]
    )
    assert all(snapshot["open_boundary"] for snapshot in analysis.observables[0]["cycles"])


def test_declared_missing_observable_builder_only_falls_back_overlay() -> None:
    def unavailable(shape: PlaquetteShape2D, cube: CubeWithArms) -> list[tuple[int, int]]:
        raise NotImplementedError("This convention has no cube readout representatives.")

    base = FIXED_BULK_CONVENTION
    convention = replace(
        base,
        triplet=replace(
            base.triplet,
            observable_builder=replace(
                base.triplet.observable_builder,
                cube_top_readouts_builder=unavailable,
            ),
        ),
    )
    graph = memory()
    computation = lower_block_graph(graph, 1, convention)
    analysis = analyse_computation(computation, graph.find_correlation_surfaces())
    assert computation.fully_supported
    assert len(analysis.cycles) == 3
    assert all(len(cycle.checks) == 8 for cycle in analysis.cycles.values())
    assert not analysis.observables[0]["cycles"]
    assert any("readout representatives" in n for n in analysis.observables[0]["notices"])


def test_unexpected_observable_builder_failure_is_not_hidden() -> None:
    def broken(shape: PlaquetteShape2D, cube: CubeWithArms) -> list[tuple[int, int]]:
        raise ValueError("Unexpected builder bug.")

    base = FIXED_BULK_CONVENTION
    convention = replace(
        base,
        triplet=replace(
            base.triplet,
            observable_builder=replace(
                base.triplet.observable_builder,
                cube_top_readouts_builder=broken,
            ),
        ),
    )
    graph = memory()
    computation = lower_block_graph(graph, 1, convention)
    with pytest.raises(ValueError, match="Unexpected builder bug"):
        analyse_computation(computation, graph.find_correlation_surfaces())


def test_split_does_not_remove_checks_that_persist_on_sibling_patch() -> None:
    graph = BlockGraph()
    for x in (0, 1):
        for z in (0, 1):
            graph.add_cube(Position3D(x, 0, z), "ZXZ")
        graph.add_pipe(Position3D(x, 0, 0), Position3D(x, 0, 1))
    graph.add_pipe(Position3D(0, 0, 0), Position3D(1, 0, 0))
    computation = lower_block_graph(graph, 1)
    analysis = analyse_computation(computation)
    previous = next(c for c in computation.cycles if c.block_z == 0 and c.round_index == 2)
    successors = [c for c in computation.cycles if c.block_z == 1 and c.round_index == 0]
    old_checks = {c.output_support for c in analysis.cycles[previous.index].checks}
    surviving_checks = {c.support for p in successors for c in analysis.cycles[p.index].checks}
    removed = [
        c.output_support for p in successors for c in analysis.cycles[p.index].removed_checks
    ]
    assert old_checks - surviving_checks
    assert len(removed) == len(set(removed))
    assert set(removed) == old_checks - surviving_checks
