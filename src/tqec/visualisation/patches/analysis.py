"""Circuit-derived checks and logical operators for the physical patch viewer.

The analysis deliberately uses the *merged* circuit. In particular a pair of
extended plaquettes need not measure two independent checks: their joint
measurement parity can be the only data-only stabilizer flow.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import stim

from tqec.circuit.qubit import GridQubit
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.observables.abstract_observable import (
    AbstractObservable,
    CubeWithArms,
    PipeWithArms,
    _check_correlation_surface_validity,
    compile_correlation_surface_to_abstract_observable,
)
from tqec.compile.observables.builder import ObservableComponent
from tqec.compile.specs.enums import SpatialArms
from tqec.computation.correlation import CorrelationSurface
from tqec.computation.cube import Cube, ZXCube
from tqec.utils.array import to2dlist
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Direction3D
from tqec.visualisation.patches.model import PatchComputation, PatchCycle

Support = tuple[tuple[int, int, str], ...]
"""Nonidentity Pauli terms in physical qubit coordinates."""


@dataclass(frozen=True)
class QubitRoles:
    """Actual data and measurement-qubit roles after pipe substitution."""

    data: frozenset[GridQubit]
    syndrome: frozenset[GridQubit]


def qubit_roles(layer: LayoutLayer, k: int) -> QubitRoles:
    """Recover roles from the fully merged template, including surgery bridges."""
    template, plaquettes = layer.to_template_and_plaquettes()
    array = template.instantiate(k)
    increment = template.get_increments()
    minimum, _ = layer.bounds
    shape = layer.element_shape.to_shape_2d(k)
    data: set[GridQubit] = set()
    syndrome: set[GridQubit] = set()
    for y, row in enumerate(to2dlist(array)):
        for x, index in enumerate(row):
            if not index:
                continue
            plaquette = plaquettes[int(index)]
            dx = plaquette.origin.x + x * increment.x + minimum.x * (shape.x - 1)
            dy = plaquette.origin.y + y * increment.y + minimum.y * (shape.y - 1)
            used = plaquette.circuit.qubits
            data.update(
                GridQubit(q.x + dx, q.y + dy) for q in plaquette.qubits.data_qubits if q in used
            )
            syndrome.update(
                GridQubit(q.x + dx, q.y + dy) for q in plaquette.qubits.syndrome_qubits if q in used
            )
    return QubitRoles(frozenset(data), frozenset(syndrome))


def _support(pauli: stim.PauliString, coordinates: dict[int, GridQubit]) -> Support:
    return tuple(
        sorted(
            (coordinates[i].x, coordinates[i].y, "_XYZ"[pauli[i]])
            for i in pauli.pauli_indices()
            if i in coordinates
        )
    )


def _phase(pauli: stim.PauliString) -> str:
    return {1: "+", -1: "-", 1j: "+i", -1j: "-i"}[pauli.sign]


def _frame_pauli(flow: stim.Flow | None, frame: Literal["input", "output"]) -> stim.PauliString:
    if flow is None:
        return stim.PauliString()
    inp, out = flow.input_copy(), flow.output_copy()
    if frame == "input":
        inp.sign *= out.sign.conjugate()
        return inp
    out.sign *= inp.sign.conjugate()
    return out


def _pauli_bits(pauli: stim.PauliString) -> int:
    bits = 0
    for i in pauli.pauli_indices():
        p = pauli[i]
        if p in (1, 2):
            bits |= 1 << (2 * i)
        if p in (2, 3):
            bits |= 1 << (2 * i + 1)
    return bits


def _measurement_bits(flow: stim.Flow) -> int:
    bits = 0
    for m in flow.measurements_copy():
        bits ^= 1 << m
    return bits


def _flow_bits(flow: stim.Flow, n: int) -> int:
    return (
        _pauli_bits(flow.input_copy())
        | (_pauli_bits(flow.output_copy()) << (2 * n))
        | (_measurement_bits(flow) << (4 * n))
    )


def _kernel(flows: list[stim.Flow], forbidden: Any) -> list[stim.Flow]:
    """Eliminate forbidden terms while preserving the exact signed flow product."""
    pivots: dict[int, tuple[int, stim.Flow]] = {}
    result: list[stim.Flow] = []
    for original in flows:
        flow = original
        bits = forbidden(flow)
        while bits:
            pivot = bits.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = bits, flow
                break
            row, other = pivots[pivot]
            bits ^= row
            flow = flow * other
        else:
            result.append(flow)
    return result


def _solve(flows: list[stim.Flow], target: int, project: Any) -> tuple[stim.Flow, list[int]] | None:
    """Solve binary constraints, retaining the indices and phases of all factors."""
    pivots: dict[int, tuple[int, stim.Flow, int]] = {}
    for i, original in enumerate(flows):
        row, flow, factors = project(original), original, 1 << i
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row, flow, factors
                break
            other_row, other_flow, other_factors = pivots[pivot]
            row ^= other_row
            flow = flow * other_flow
            factors ^= other_factors
    flow = stim.Flow()
    factors = 0
    while target:
        pivot = target.bit_length() - 1
        if pivot not in pivots:
            return None
        row, other_flow, other_factors = pivots[pivot]
        target ^= row
        flow = flow * other_flow
        factors ^= other_factors
    return flow, [i for i in range(len(flows)) if factors & (1 << i)]


@dataclass(frozen=True)
class MeasuredCheck:
    """A measured data stabilizer, possibly requiring several syndrome outcomes."""

    support: Support
    output_support: Support
    measurements: tuple[int, ...]
    sign: str = "+"
    output_sign: str = "+"
    input_flow: stim.Flow | None = field(default=None, compare=False, repr=False)
    output_flow: stim.Flow | None = field(default=None, compare=False, repr=False)

    def to_dict(self, index: int = 0, added: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible check, with measurement indices local to the round."""
        return {
            "id": index,
            "support": self.support or self.output_support,
            "input_support": self.support,
            "output_support": self.output_support,
            "measurements": self.measurements,
            "sign": self.sign,
            "output_sign": self.output_sign,
            "added": added,
            "frame": "input" if self.support else "output",
        }


@dataclass
class CycleAnalysis:
    """Cached physical analysis of one concrete round."""

    roles: QubitRoles
    checks: list[MeasuredCheck]
    diagnostic_circuit: stim.Circuit
    diagnostic_to_original_measurements: tuple[int, ...]
    added: set[Support] = field(default_factory=set)
    removed_checks: list[MeasuredCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return check and qubit records consumed by the viewer."""
        return {
            "qubits": [
                {"x": q.x, "y": q.y, "role": role}
                for role, qubits in [("data", self.roles.data), ("syndrome", self.roles.syndrome)]
                for q in sorted(qubits)
            ],
            "checks": [
                c.to_dict(i, (c.support or c.output_support) in self.added)
                for i, c in enumerate(self.checks)
            ],
            "removed_checks": [
                c.to_dict(i)
                | {"support": c.output_support, "sign": c.output_sign, "frame": "output"}
                for i, c in enumerate(self.removed_checks)
            ],
        }


_DATA_BOUNDARIES = {"R", "RX", "RY", "M", "MX", "MY", "MR", "MRX", "MRY"}
_MEASUREMENTS = {"M", "MX", "MY", "MR", "MRX", "MRY"}


def _diagnostic_circuit(
    circuit: stim.Circuit, data_indices: set[int]
) -> tuple[stim.Circuit, tuple[int, ...]]:
    """Remove only data preparation/readout; retain syndrome resets and Cliffords."""
    result = stim.Circuit()
    measurement_map: list[int] = []
    measurement_index = 0
    for instruction in circuit:
        assert isinstance(instruction, stim.CircuitInstruction)
        if instruction.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        targets = instruction.targets_copy()
        if instruction.name in _DATA_BOUNDARIES:
            kept = []
            for target in targets:
                if target.value not in data_indices:
                    kept.append(target)
                    if instruction.name in _MEASUREMENTS:
                        measurement_map.append(measurement_index)
                if instruction.name in _MEASUREMENTS:
                    measurement_index += 1
            if kept:
                result.append(instruction.name, kept, instruction.gate_args_copy())
        else:
            result.append(instruction)
            count = stim.Circuit(str(instruction)).num_measurements
            measurement_map.extend(range(measurement_index, measurement_index + count))
            measurement_index += count
    return result, tuple(measurement_map)


def extract_checks(
    circuit: stim.Circuit, coordinates: dict[int, GridQubit], data: frozenset[GridQubit]
) -> tuple[list[MeasuredCheck], stim.Circuit, tuple[int, ...]]:
    """Extract independent complete checks in both frames by signed flow elimination.

    Single-syndrome outcomes are preferred. If extended ancilla circuitry only
    measures a joint parity, the remaining parity basis exposes the complete
    check instead of incorrectly displaying its individual plaquette halves.
    """
    data_indices = {i for i, q in coordinates.items() if q in data}
    diagnostic, measurement_map = _diagnostic_circuit(circuit, data_indices)
    generators = diagnostic.flow_generators()
    n = diagnostic.num_qubits
    ancilla_mask = sum(3 << (2 * i) for i in range(n) if i not in data_indices)
    by_frame: dict[str, dict[int, stim.Flow]] = {}
    for frame in ("input", "output"):

        def forbidden(flow: stim.Flow) -> int:
            inp, out = _pauli_bits(flow.input_copy()), _pauli_bits(flow.output_copy())
            if frame == "input":
                return out | ((inp & ancilla_mask) << (2 * n))
            return inp | ((out & ancilla_mask) << (2 * n))

        candidates = _kernel(generators, forbidden)
        selected: list[stim.Flow] = []
        for m in range(diagnostic.num_measurements):
            solution = _solve(candidates, 1 << m, _measurement_bits)
            if solution is not None:
                selected.append(solution[0])
        # Add independent joint parities after individual outcomes.
        parity_basis: dict[int, int] = {}
        independent: dict[int, stim.Flow] = {}
        for flow in selected + candidates:
            parity = _measurement_bits(flow)
            row = parity
            while row:
                pivot = row.bit_length() - 1
                if pivot not in parity_basis:
                    parity_basis[pivot] = row
                    independent[parity] = flow
                    break
                row ^= parity_basis[pivot]
        by_frame[frame] = independent
    checks: list[MeasuredCheck] = []
    for parity in sorted(set(by_frame["input"]) | set(by_frame["output"])):
        input_flow = by_frame["input"].get(parity)
        output_flow = by_frame["output"].get(parity)
        # Bases chosen independently can differ: recover the corresponding
        # parity rather than mistaking a change of basis for a missing frame.
        if input_flow is None:
            solved = _solve(list(by_frame["input"].values()), parity, _measurement_bits)
            input_flow = solved[0] if solved else None
        if output_flow is None:
            solved = _solve(list(by_frame["output"].values()), parity, _measurement_bits)
            output_flow = solved[0] if solved else None
        inp = _frame_pauli(input_flow, "input")
        out = _frame_pauli(output_flow, "output")
        if not inp.weight and not out.weight:
            continue
        checks.append(
            MeasuredCheck(
                _support(inp, coordinates),
                _support(out, coordinates),
                tuple(measurement_map[i] for i in range(len(measurement_map)) if parity & (1 << i)),
                _phase(inp),
                _phase(out),
                input_flow,
                output_flow,
            )
        )
    return checks, diagnostic, measurement_map


def analyse_cycle(cycle: PatchCycle, k: int) -> CycleAnalysis:
    """Analyse merged geometry and measured stabilizers for one cycle."""
    roles = qubit_roles(cycle.layer, k)
    checks, diagnostic, mapping = extract_checks(
        cycle.circuit.get_circuit(include_qubit_coords=False),
        dict(cycle.circuit.qubit_map.items()),
        roles.data,
    )
    return CycleAnalysis(roles, checks, diagnostic, mapping)


@dataclass
class AnalysisResult:
    """Physical analysis and independently available logical traces."""

    cycles: dict[int, CycleAnalysis]
    observables: list[dict[str, Any]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


@dataclass
class _Snapshot:
    cycle: PatchCycle
    tick: int
    instruction: int
    measurement: int
    local_measurements: dict[int, int]


def _split_snapshot(
    circuit: stim.Circuit, data_indices: set[int]
) -> tuple[stim.Circuit, stim.Circuit, dict[int, int]]:
    """Place the cut before final data readout, preserving record identities.

    Operations in the final scheduled moment are disjoint. Only its data
    readouts move past the cut; syndrome readouts stay before it.
    """
    instructions = [circuit[i] for i in range(len(circuit))]
    last_tick = max(
        (i for i, instruction in enumerate(instructions) if instruction.name == "TICK"),
        default=-1,
    )
    before = stim.Circuit()
    after = stim.Circuit()
    before_records: list[int] = []
    after_records: list[int] = []
    original_index = 0
    for i, instruction in enumerate(instructions):
        assert isinstance(instruction, stim.CircuitInstruction)
        if instruction.name in _MEASUREMENTS:
            early: list[stim.GateTarget] = []
            late: list[stim.GateTarget] = []
            for target in instruction.targets_copy():
                if target.value in data_indices:
                    if i <= last_tick:
                        raise NotImplementedError("Data readout before the final scheduled moment.")
                    late.append(target)
                    after_records.append(original_index)
                else:
                    early.append(target)
                    before_records.append(original_index)
                original_index += 1
            if early:
                before.append(instruction.name, early, instruction.gate_args_copy())
            if late:
                after.append(instruction.name, late, instruction.gate_args_copy())
        else:
            before.append(instruction)
            count = stim.Circuit(str(instruction)).num_measurements
            before_records.extend(range(original_index, original_index + count))
            original_index += count
    mapping = {old: new for new, old in enumerate(before_records + after_records)}
    return before, after, mapping


def _measurement_qubits(circuit: stim.Circuit) -> dict[int, int]:
    """Map each measured qubit to its final original measurement index."""
    records: dict[int, int] = {}
    count = 0
    for instruction in circuit:
        assert isinstance(instruction, stim.CircuitInstruction)
        if instruction.name in _MEASUREMENTS:
            for target in instruction.targets_copy():
                records[target.value] = count
                count += 1
        else:
            count += stim.Circuit(str(instruction)).num_measurements
    return records


def _build_snapshot_circuit(
    computation: PatchComputation,
    analyses: dict[int, CycleAnalysis],
    abstract_observables: list[AbstractObservable],
) -> tuple[stim.Circuit, list[_Snapshot], dict[int, GridQubit], list[set[int]]]:
    qubits = sorted({q for cycle in computation.cycles for q in cycle.circuit.qubits})
    indices = {q: i for i, q in enumerate(qubits)}
    coordinates = dict(enumerate(qubits))
    circuit = stim.Circuit()
    snapshots: list[_Snapshot] = []
    observable_measurements: list[set[int]] = [set() for _ in abstract_observables]
    d = computation.distance
    groups: dict[tuple[int, int], list[PatchCycle]] = defaultdict(list)
    for cycle in computation.cycles:
        groups[cycle.block_z, cycle.round_index].append(cycle)
    for group in groups.values():
        pending = []
        for cycle in group:
            mapped = cycle.circuit.map_qubit_indices(
                {i: indices[q] for i, q in cycle.circuit.qubit_map.items()}
            ).get_circuit(include_qubit_coords=False)
            before, after, mapping = _split_snapshot(
                mapped, {indices[q] for q in analyses[cycle.index].roles.data}
            )
            base = circuit.num_measurements
            circuit += before
            pending.append((cycle, mapped, after, mapping, base, before.num_measurements))
        cut = circuit.num_ticks, len(circuit), circuit.num_measurements
        circuit.append("TICK", [])
        completed = []
        for cycle, mapped, after, local_mapping, base, num_before in pending:
            base_after = circuit.num_measurements
            mapping = {
                old: base + new if new < num_before else base_after + new - num_before
                for old, new in local_mapping.items()
            }
            snapshots.append(_Snapshot(cycle, *cut, mapping))
            circuit += after
            completed.append((cycle, mapped, mapping))
        for cycle, mapped, mapping in completed:
            if not abstract_observables:
                continue
            template, _ = cycle.layer.to_template_and_plaquettes()
            records = _measurement_qubits(mapped)
            components = []
            if cycle.round_index == 0:
                components.append(ObservableComponent.BOTTOM_STABILIZERS)
            if cycle.round_index == d - 1:
                components.append(ObservableComponent.TOP_READOUTS)
            if cycle.is_transition:
                components.append(ObservableComponent.REALIGNMENT)
            for obs_id, observable in enumerate(abstract_observables):
                included: set[int] = set()
                for component in components:
                    selected = computation.convention.triplet.observable_builder.build(
                        computation.k, template, observable.slice_at_z(cycle.block_z), component
                    )
                    for q in selected:
                        if q in indices and indices[q] in records:
                            included.symmetric_difference_update([mapping[records[indices[q]]]])
                if included:
                    observable_measurements[obs_id].symmetric_difference_update(included)
                    circuit.append(
                        "OBSERVABLE_INCLUDE",
                        [stim.target_rec(m - circuit.num_measurements) for m in sorted(included)],
                        obs_id,
                    )
        circuit.append("TICK", [])
    return circuit, snapshots, coordinates, observable_measurements


def _pauli_from_support(support: Support, coordinates: dict[int, GridQubit]) -> stim.PauliString:
    indices = {(q.x, q.y): i for i, q in coordinates.items()}
    result = stim.PauliString(max(coordinates, default=-1) + 1)
    for x, y, basis in support:
        result[indices[x, y]] = basis
    return result


def explain_transition(
    segment: stim.Circuit,
    previous: stim.PauliString,
    current: stim.PauliString,
    measurements: list[int],
    coordinates: dict[int, GridQubit],
    checks: list[tuple[MeasuredCheck, list[int]]],
) -> dict[str, Any] | None:
    """Validate and factor a logical transition using signed circuit flows.

    Physical check factors are preferred over generic preparation/readout
    factors. The latter are named explicitly and are never presented as round
    stabilizers. All indices in ``measurements`` and ``checks`` are local to
    ``segment``.
    """
    desired = stim.Flow(input=previous, output=current, measurements=measurements)
    if not segment.has_flow(desired):
        current = -current
        desired = stim.Flow(input=previous, output=current, measurements=measurements)
        if not segment.has_flow(desired):
            return None
    transport = stim.Flow(input=previous, output=previous)
    generators: list[stim.Flow] | None = None
    if not segment.has_flow(transport):
        generators = segment.flow_generators()
        n = max(segment.num_qubits, len(previous), len(current))
        # Prefer unchanged support allowing a measured parity, then let the
        # actual Clifford circuit determine the transported representative.
        mask = (1 << (4 * n)) - 1
        solved = _solve(
            generators, _flow_bits(transport, n) & mask, lambda f: _flow_bits(f, n) & mask
        )
        if solved is None:
            solved = _solve(
                generators, _pauli_bits(previous), lambda f: _pauli_bits(f.input_copy())
            )
        if solved is None:
            return None
        transport = solved[0]
    residual = desired * transport
    n = max(segment.num_qubits, len(previous), len(current))
    physical_flows: list[stim.Flow] = []
    physical_factors: list[dict[str, Any]] = []
    for check, parity in checks:
        if not check.output_support:
            continue
        output = _pauli_from_support(check.output_support, coordinates)
        if check.output_sign == "-":
            output = -output
        candidate = stim.Flow(output=output, measurements=parity)
        if segment.has_flow(candidate):
            physical_flows.append(candidate)
            physical_factors.append(
                {
                    "support": check.output_support,
                    "measurements": parity,
                    "sign": check.output_sign,
                    "label": "Measured stabilizer",
                }
            )
    factor_flows = physical_flows.copy()
    factor_records = physical_factors.copy()
    target = _flow_bits(residual, n)
    solved = _solve(factor_flows, target, lambda f: _flow_bits(f, n))
    if solved is None:
        if generators is None:
            generators = segment.flow_generators()
        boundary = _kernel(generators, lambda f: _pauli_bits(f.input_copy()))
        factor_flows.extend(boundary)
        factor_records.extend(
            {
                "support": _support(f.output_copy(), coordinates),
                "measurements": f.measurements_copy(),
                "sign": _phase(_frame_pauli(f, "output")),
                "label": "Preparation/readout or post-round stabilizer",
            }
            for f in boundary
        )
        solved = _solve(factor_flows, target, lambda f: _flow_bits(f, n))
    if solved is None:
        return None
    product, factor_indices = solved
    # Binary elimination does not discard signs: validate the reconstructed
    # signed relation as well as the desired relation against the segment.
    if (
        _flow_bits(product, n) != _flow_bits(residual, n)
        or _frame_pauli(product, "output").sign != _frame_pauli(residual, "output").sign
        or not segment.has_flow(product)
        or not segment.has_flow(transport * product)
    ):
        return None
    pure = transport == stim.Flow(input=previous, output=previous)
    return {
        "validated": True,
        "kind": "stabilizer_product" if pure else "circuit_transport",
        "factors": [factor_records[i] for i in factor_indices],
        "measurements": measurements,
        "sign": _phase(current),
        "transport_support": _support(transport.output_copy(), coordinates),
        "transport_measurements": transport.measurements_copy(),
        "description": (
            "Logical representatives differ by the shown stabilizer product."
            if pure and factor_indices
            else "Logical operator is unchanged."
            if pure
            else "Validated circuit transport, including Clifford and boundary contributions."
        ),
    }


def _full_logical_traces(
    computation: PatchComputation,
    analyses: dict[int, CycleAnalysis],
    surfaces: list[CorrelationSurface],
) -> list[dict[str, Any]]:
    abstract = [
        compile_correlation_surface_to_abstract_observable(
            computation.graph.fix_shadowed_faces(),
            surface,
            computation.convention.name == "fixed_bulk",
        )
        for surface in surfaces
    ]
    circuit, snapshots, coordinates, contributions = _build_snapshot_circuit(
        computation, analyses, abstract
    )
    regions = circuit.detecting_regions(targets=["L"], ticks=[s.tick for s in snapshots])
    result: list[dict[str, Any]] = []
    for obs_id, surface in enumerate(surfaces):
        trace: dict[str, Any] = {"name": f"Observable {obs_id}", "cycles": [], "notices": []}
        sensitivities = regions.get(stim.target_logical_observable_id(obs_id), {})
        previous: _Snapshot | None = None
        previous_pauli: stim.PauliString | None = None
        transitions: dict[int, dict[str, Any] | None] = {}
        for snapshot in snapshots:
            cycle = snapshot.cycle
            pauli = sensitivities.get(snapshot.tick, stim.PauliString(circuit.num_qubits))
            allowed = analyses[cycle.index].roles.data | analyses[cycle.index].roles.syndrome
            local_coordinates = {i: q for i, q in coordinates.items() if q in allowed}
            support = _support(pauli, local_coordinates)
            record: dict[str, Any] = {
                "cycle_id": cycle.index,
                "support": support,
                "factors": [],
                "description": "Logical snapshot after syndrome extraction, before data readout.",
                "validated": True,
            }
            if previous is None and not pauli.weight:
                prefix = circuit[: snapshot.instruction]
                parity = [m for m in sorted(contributions[obs_id]) if m < snapshot.measurement]
                boundary_flows: list[stim.Flow] = []
                boundary_factors: list[dict[str, Any]] = []
                for peer in snapshots:
                    if peer.instruction != snapshot.instruction:
                        continue
                    for check in analyses[peer.cycle.index].checks:
                        if not check.output_support:
                            continue
                        output = _pauli_from_support(check.output_support, coordinates)
                        if check.output_sign == "-":
                            output = -output
                        measurements = [peer.local_measurements[m] for m in check.measurements]
                        candidate = stim.Flow(output=output, measurements=measurements)
                        if prefix.has_flow(candidate):
                            boundary_flows.append(candidate)
                            boundary_factors.append(
                                {
                                    "support": check.output_support,
                                    "measurements": measurements,
                                    "sign": check.output_sign,
                                    "label": "Initial observable stabilizer",
                                }
                            )
                target = sum(1 << m for m in parity)
                solved = _solve(boundary_flows, target, _measurement_bits)
                if (
                    parity
                    and solved is not None
                    and prefix.has_flow(stim.Flow(measurements=parity))
                ):
                    record.update(
                        {
                            "kind": "measurement_boundary",
                            "factors": [boundary_factors[i] for i in solved[1]],
                            "measurements": parity,
                            "description": "Observable parity is the shown stabilizer product, "
                            "fixed by initial data preparations.",
                        }
                    )
                    transitions[snapshot.instruction] = {
                        key: value
                        for key, value in record.items()
                        if key not in {"support", "cycle_id"}
                    }
            if snapshot.instruction in transitions:
                transition = transitions[snapshot.instruction]
                if transition is not None:
                    record.update(transition)
            elif previous is not None and previous_pauli is not None:
                segment = circuit[previous.instruction : snapshot.instruction]
                parity = [
                    m - previous.measurement
                    for m in sorted(contributions[obs_id])
                    if previous.measurement <= m < snapshot.measurement
                ]
                check_flows = [
                    (
                        check,
                        [
                            peer.local_measurements[m] - previous.measurement
                            for m in check.measurements
                        ],
                    )
                    for peer in snapshots
                    if peer.instruction == snapshot.instruction
                    for check in analyses[peer.cycle.index].checks
                    if all(
                        previous.measurement <= peer.local_measurements[m] < snapshot.measurement
                        for m in check.measurements
                    )
                ]
                transition = explain_transition(
                    segment, previous_pauli, pauli, parity, coordinates, check_flows
                )
                if transition is not None:
                    record.update(transition)
                else:
                    record["description"] = (
                        "Logical snapshot established; transition factorization unavailable."
                    )
                transitions[snapshot.instruction] = transition
            local_xy = {(q.x, q.y) for q in allowed}
            record["factors"] = [
                factor
                for factor in record["factors"]
                if any((x, y) in local_xy for x, y, _ in factor["support"])
            ]
            # Empty support is still an established physical trace, not a
            # request to insert a topological operator into this cycle.
            trace["cycles"].append(record)
            transitions.setdefault(snapshot.instruction, None)
            if record.get("sign") == "-":
                pauli = -pauli
            previous, previous_pauli = snapshot, pauli
        result.append(trace)
    return result


def _local_representative(
    computation: PatchComputation,
    cycle: PatchCycle,
    surface: CorrelationSurface,
    coordinates: dict[int, GridQubit],
) -> stim.PauliString | None:
    """Lower open surface ends to convention-specific patch representatives.

    This does not cap a fragment with fictitious preparations or measurements.
    Its returned operator still has to pass local circuit-flow validation.
    """
    template, _ = cycle.layer.to_template_and_plaquettes()
    builder = computation.convention.triplet.observable_builder
    indices = {q: i for i, q in coordinates.items()}
    pauli = stim.PauliString(max(coordinates, default=-1) + 1)
    shape = template.element_shape(computation.k)
    for position in cycle.component:
        if position not in surface.positions:
            continue
        cube = computation.graph[position]
        if not isinstance(cube.kind, ZXCube):
            return None
        for basis in sorted(surface.bases_at(position), key=lambda b: b.value):
            arms = SpatialArms.NONE
            if cube.is_spatial:
                for arm, shift in SpatialArms.get_map_from_arm_to_shift().items():
                    neighbour = position.shift_by(*shift)
                    if any(
                        edge.get_basis(position) == basis
                        and neighbour in (edge.u.position, edge.v.position)
                        for edge in surface.edges_at(position)
                    ):
                        arms |= arm
                if basis != cube.kind.z or len(arms) != 2:
                    return None
                representative_cube = cube
            else:
                representative_cube = Cube(
                    position, cube.kind.with_basis_along(Direction3D.Z, basis)
                )
            local = builder.cube_top_readouts_builder(
                shape, CubeWithArms(representative_cube, arms)
            )
            selected = builder.transform_coords_into_grid(computation.k, template, local, position)
            if not selected <= indices.keys():
                return None
            factor = stim.PauliString(len(pauli))
            for q in selected:
                factor[indices[q]] = basis.value
            pauli *= factor
    component = set(cycle.component)
    for pipe in computation.graph.pipes:
        if pipe.u.position not in component or pipe.v.position not in component:
            continue
        if pipe.direction == Direction3D.Z:
            continue
        matching = [
            edge
            for edge in surface.span
            if {edge.u.position, edge.v.position} == {pipe.u.position, pipe.v.position}
            and edge.u.basis == pipe.kind.z
        ]
        if not matching:
            continue
        arms = tuple(
            SpatialArms.from_cube_in_graph(c, computation.graph)
            if c.is_spatial
            else SpatialArms.NONE
            for c in pipe
        )
        local = builder.pipe_top_readouts_builder(shape, PipeWithArms(pipe, (arms[0], arms[1])))
        selected = builder.transform_coords_into_grid(
            computation.k, template, local, pipe.u.position
        )
        factor = stim.PauliString(len(pauli))
        for q in selected:
            if q in indices:
                factor[indices[q]] = matching[0].u.basis.value
        pauli *= factor
    pauli.sign = 1
    return pauli


def _hybrid_logical_traces(
    computation: PatchComputation,
    analyses: dict[int, CycleAnalysis],
    surfaces: list[CorrelationSurface],
) -> list[dict[str, Any]]:
    # Each contiguous component fragment is independent. No circuit is joined
    # across an unavailable operation, even when its endpoints look identical.
    result: list[dict[str, Any]] = []
    for obs_id, surface in enumerate(surfaces):
        trace: dict[str, Any] = {"name": f"Observable {obs_id}", "cycles": [], "notices": []}
        previous: dict[tuple[Any, ...], tuple[PatchCycle, stim.PauliString, stim.Circuit]] = {}
        for cycle in computation.cycles:
            if not set(cycle.component) & surface.positions:
                continue
            coordinates = dict(cycle.circuit.qubit_map.items())
            analysis = analyses[cycle.index]
            try:
                pauli = _local_representative(computation, cycle, surface, coordinates)
            except NotImplementedError as exc:
                notice = f"Local observable lowering is unavailable: {exc}"
                if notice not in trace["notices"]:
                    trace["notices"].append(notice)
                continue
            if pauli is None or not pauli.weight:
                continue
            # A candidate must commute with every complete measured check and
            # must be carried by the actual syndrome-extraction circuit.
            checks = [
                _pauli_from_support(c.output_support, coordinates)
                for c in analysis.checks
                if c.output_support
            ]
            if any(not pauli.commutes(check) for check in checks):
                continue
            diagnostic = analysis.diagnostic_circuit
            flow = stim.Flow(input=pauli, output=pauli)
            if not diagnostic.has_flow(flow):
                continue
            try:
                before, after, mapping = _split_snapshot(
                    cycle.circuit.get_circuit(False),
                    {i for i, q in coordinates.items() if q in analysis.roles.data},
                )
            except NotImplementedError as exc:
                notice = f"Local logical snapshot is unavailable: {exc}"
                if notice not in trace["notices"]:
                    trace["notices"].append(notice)
                continue
            record: dict[str, Any] = {
                "cycle_id": cycle.index,
                "support": _support(pauli, coordinates),
                "factors": [],
                "validated": True,
                "open_boundary": True,
                "description": "Locally verified logical operator with an open temporal boundary.",
            }
            # Qubit indices can change between different concrete layouts.
            # Rebase the previous operator and readout tail before joining.
            key = cycle.component
            last = previous.get(key)
            if last is not None and last[0].round_index + 1 == cycle.round_index:
                old_cycle, old_pauli, old_tail = last
                old_coordinates = dict(old_cycle.circuit.qubit_map.items())
                if old_coordinates == coordinates:
                    segment = old_tail + before
                    check_flows = [
                        (
                            check,
                            [old_tail.num_measurements + mapping[m] for m in check.measurements],
                        )
                        for check in analysis.checks
                    ]
                    explanation = explain_transition(
                        segment, old_pauli, pauli, [], coordinates, check_flows
                    )
                    if explanation is not None:
                        record.update(explanation)
            trace["cycles"].append(record)
            previous[key] = cycle, pauli, after
        if len(trace["cycles"]) < len(computation.cycles):
            trace["notices"].append(
                "Physical logical tracking is local; unverified segments retain "
                "their topological surface."
            )
        result.append(trace)
    return result


def _mark_stabilizer_changes(
    computation: PatchComputation, analyses: dict[int, CycleAnalysis]
) -> None:
    """Compare physical supports only at adjacent, available component rounds."""
    by_time: dict[tuple[int, int], list[PatchCycle]] = defaultdict(list)
    for cycle in computation.cycles:
        by_time[cycle.block_z, cycle.round_index].append(cycle)
    for cycle in computation.cycles:
        if cycle.round_index:
            previous = [
                c
                for c in by_time[cycle.block_z, cycle.round_index - 1]
                if set(c.component) & set(cycle.component)
            ]
        else:
            previous_positions = {
                p.shift_by(dz=-1)
                for p in cycle.component
                if computation.graph.has_pipe_between(p.shift_by(dz=-1), p)
            }
            previous_by_component = {
                c.component: c
                for c in by_time[cycle.block_z - 1, computation.distance - 1]
                if set(c.component) & previous_positions
            }
            # Replace each component separately. A realignment at one source
            # must not hide the normal last round of another source.
            previous_by_component.update(
                {
                    c.component: c
                    for c in by_time[cycle.block_z - 1, computation.distance]
                    if set(c.component) & previous_positions
                }
            )
            previous = list(previous_by_component.values())
        if not previous:
            continue
        current_analysis = analyses[cycle.index]
        previous_checks = {
            c.output_support: c
            for prev in previous
            for c in analyses[prev.index].checks
            if c.output_support
        }
        current_checks = {c.support: c for c in current_analysis.checks if c.support}
        current_analysis.added = (set(current_checks) - set(previous_checks)) | {
            c.output_support for c in current_analysis.checks if not c.support and c.output_support
        }
        # A split has several successor components. Checks that persist on a
        # sibling patch have not been removed from the computation. Compare
        # against all directly connected successors, and display an actually
        # removed check once, on the first successor touching its support.
        previous_positions = {p for prev in previous for p in prev.component}
        successors = [
            peer
            for peer in by_time[cycle.block_z, cycle.round_index]
            if any(
                p in previous_positions
                if cycle.round_index
                else p.shift_by(dz=-1) in previous_positions
                and computation.graph.has_pipe_between(p.shift_by(dz=-1), p)
                for p in peer.component
            )
        ]
        successor_checks = {
            check.support
            for peer in successors
            for check in analyses[peer.index].checks
            if check.support
        }

        def removal_owner(support: Support) -> int:
            support_qubits = {GridQubit(x, y) for x, y, _ in support}
            return min(
                (
                    peer.index
                    for peer in successors
                    if support_qubits & analyses[peer.index].roles.data
                ),
                default=min(peer.index for peer in successors),
            )

        current_analysis.removed_checks = [
            previous_checks[s]
            for s in sorted(set(previous_checks) - successor_checks)
            if removal_owner(s) == cycle.index
        ]


def analyse_computation(
    computation: PatchComputation,
    surfaces: list[CorrelationSurface] | None = None,
) -> AnalysisResult:
    """Analyse all known rounds and independently lower each logical overlay.

    Repeated rounds share extracted checks, but change annotations are per
    occurrence. Unavailable intervals never acquire artificial circuit flows.
    """
    cache: dict[tuple[str, QubitRoles], CycleAnalysis] = {}
    cycles: dict[int, CycleAnalysis] = {}
    for cycle in computation.cycles:
        roles = qubit_roles(cycle.layer, computation.k)
        key = str(cycle.circuit.get_circuit()), roles
        if key not in cache:
            cache[key] = analyse_cycle(cycle, computation.k)
        cached = cache[key]
        cycles[cycle.index] = CycleAnalysis(
            cached.roles,
            cached.checks,
            cached.diagnostic_circuit,
            cached.diagnostic_to_original_measurements,
        )
    _mark_stabilizer_changes(computation, cycles)
    result = AnalysisResult(cycles)
    if not surfaces:
        return result
    for obs_id, surface in enumerate(surfaces):
        # Missing objects are malformed input, while parity-open ends are a
        # legitimate partial surface and can be lowered locally.
        for position in surface.positions:
            if position not in computation.graph:
                raise TQECError(f"Correlation surface references missing cube at {position}.")
        for edge in surface.span:
            if not edge.is_self_loop and not computation.graph.has_pipe_between(
                edge.u.position, edge.v.position
            ):
                raise TQECError("Correlation surface references a missing pipe.")
        partial = not computation.fully_supported
        if not partial:
            if surface.is_single_node:
                cube = computation.graph[next(iter(surface.positions))]
                if isinstance(cube.kind, ZXCube):
                    expected_basis = cube.kind.x if cube.is_spatial else cube.kind.z
                    partial = len(computation.graph.cubes) != 1 or surface.bases_at(
                        cube.position
                    ) != {expected_basis}
                else:
                    partial = True
            elif not surface.is_single_node:
                try:
                    _check_correlation_surface_validity(surface, computation.graph.to_zx_graph())
                except TQECError:
                    partial = True
        if partial:
            observable = _hybrid_logical_traces(computation, cycles, [surface])[0]
            observable["notices"].append(
                "This surface has open or unavailable boundaries; "
                "its physical operators are verified locally."
            )
        else:
            try:
                observable = _full_logical_traces(computation, cycles, [surface])[0]
            except NotImplementedError as exc:
                # Builders explicitly declare unsupported capabilities this way.
                # Keep the geometry, try independently validated local operators,
                # and let the renderer retain topology wherever those are absent.
                observable = _hybrid_logical_traces(computation, cycles, [surface])[0]
                observable["notices"].append(f"Full observable lowering is unavailable: {exc}")
        observable["name"] = f"Observable {obs_id}"
        result.observables.append(observable)
    return result
