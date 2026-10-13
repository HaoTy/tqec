"""Compile the supported portions of a graph without inventing boundary circuits."""

from __future__ import annotations

from collections import defaultdict

from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.compile.blocks.block import Block
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.positioning import LayoutPosition2D
from tqec.compile.compile import (
    _DEFAULT_BLOCK_REPETITIONS,
    _DEFAULT_SCALABLE_QUBIT_SHAPE,
    _get_template_from_layer,
)
from tqec.compile.convention import FIXED_BULK_CONVENTION, Convention
from tqec.compile.graph import TopologicalComputationGraph
from tqec.compile.specs.base import CubeSpec, PipeSpec
from tqec.compile.specs.library.fixed_bulk import FixedBulkPipeBuilder
from tqec.compile.specs.library.generators.schedules import DIAGONAL_SCHEDULE_FAMILY
from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import Cube, ZXCube
from tqec.computation.pipe import Pipe
from tqec.utils.exceptions import TQECError
from tqec.utils.position import BlockPosition3D, Direction3D, Position3D
from tqec.visualisation.patches.model import CoarseRegion, PatchComputation, PatchCycle


def _position(pos: Position3D) -> BlockPosition3D:
    return BlockPosition3D(pos.x, pos.y, pos.z)


def _spatial_components(graph: BlockGraph) -> list[tuple[Position3D, ...]]:
    remaining = {cube.position for cube in graph.cubes}
    components: list[tuple[Position3D, ...]] = []
    while remaining:
        pending = [min(remaining)]
        component: set[Position3D] = set()
        while pending:
            pos = pending.pop()
            if pos in component:
                continue
            component.add(pos)
            remaining.remove(pos)
            for pipe in graph.pipes_at(pos):
                if pipe.kind.is_spatial:
                    other = pipe.v.position if pipe.u.position == pos else pipe.u.position
                    if other not in component:
                        pending.append(other)
        components.append(tuple(sorted(component)))
    return sorted(components, key=lambda c: (c[0].z, c))


def _extended_slices(graph: BlockGraph) -> frozenset[int]:
    def both(cube: Cube) -> bool:
        return {p.direction for p in graph.pipes_at(cube.position) if p.kind.is_spatial} == {
            Direction3D.X,
            Direction3D.Y,
        }

    return frozenset(
        pipe.u.position.z
        for pipe in graph.pipes
        if pipe.direction == Direction3D.Y and (both(pipe.u) ^ both(pipe.v))
    )


def _align_parallel_measurements(cycles: list[PatchCycle], k: int) -> None:
    """Keep component circuits on the merged compiler's common measurement moment."""
    parallel: dict[tuple[int, int], list[PatchCycle]] = defaultdict(list)
    for cycle in cycles:
        parallel[cycle.block_z, cycle.round_index].append(cycle)
    cache: dict[tuple[int, ...], list[tuple[LayoutLayer, ScheduledCircuit]]] = {}
    for group in parallel.values():
        if len({c.circuit.schedule.max_schedule for c in group}) <= 1:
            continue
        key = tuple(id(c.layer) for c in group)
        if key not in cache:
            layout = LayoutLayer(
                {pos: layer for cycle in group for pos, layer in cycle.layer.layers.items()},
                _DEFAULT_SCALABLE_QUBIT_SHAPE,
            )
            layout.reschedule_measurements()
            aligned = [
                LayoutLayer(
                    {pos: layout.layers[pos] for pos in cycle.layer.layers},
                    _DEFAULT_SCALABLE_QUBIT_SHAPE,
                )
                for cycle in group
            ]
            cache[key] = [(layer, layer.to_circuit(k)) for layer in aligned]
        for cycle, (layer, circuit) in zip(group, cache[key]):
            cycle.layer, cycle.circuit = layer, circuit


def lower_block_graph(
    graph: BlockGraph,
    k: int,
    convention: Convention = FIXED_BULK_CONVENTION,
) -> PatchComputation:
    """Lower supported components, retaining explicit coarse regions at their boundaries.

    Specifications always use the complete graph. Only the convention builders'
    explicit ``NotImplementedError`` capability declarations trigger fallback;
    validation errors and unexpected implementation failures propagate normally.
    No detector annotations, database access, or source mutations are performed.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise TQECError("k must be a positive integer.")
    graph.validate()
    result = PatchComputation(graph, convention, k)
    if not graph.cubes:
        result.notices.append("The graph is empty.")
        return result
    working = graph.fix_shadowed_faces()
    components = _spatial_components(working)
    component_at = {pos: component for component in components for pos in component}
    extended = _extended_slices(working)
    specs = {cube: CubeSpec.from_cube(cube, working, extended) for cube in working.cubes}
    cubes: dict[Position3D, Block] = {}
    coarse: dict[tuple[Position3D, ...], list[str]] = defaultdict(list)
    for cube in working.cubes:
        if not isinstance(cube.kind, ZXCube):
            coarse[component_at[cube.position]].append(
                f"{cube.kind} at {cube.position} has no physical cube implementation."
            )
            continue
        try:
            cubes[cube.position] = convention.triplet.cube_builder(
                specs[cube], _DEFAULT_BLOCK_REPETITIONS
            )
        except NotImplementedError as exc:
            coarse[component_at[cube.position]].append(
                f"Cube at {cube.position} is unsupported by {convention.name}: {exc}"
            )

    def build_pipe(pipe: Pipe, at_hadamard: bool = False) -> Block:
        builder = convention.triplet.pipe_builder
        if (
            type(builder) is FixedBulkPipeBuilder
            and pipe.kind.is_spatial
            and pipe.kind.has_hadamard
            and builder._generator._schedule_family == DIAGONAL_SCHEDULE_FAMILY
        ):
            # This preset's spatial Hadamard schedule is explicitly unsupported
            # (also recorded in the strict compiler tests). Detect it before a
            # conflicting circuit is generated; do not hide arbitrary circuit errors.
            raise NotImplementedError("The diagonal schedule does not support spatial Hadamards.")
        return convention.triplet.pipe_builder(
            PipeSpec(
                (specs[pipe.u], specs[pipe.v]),
                (
                    _get_template_from_layer(cubes[pipe.u.position]),
                    _get_template_from_layer(cubes[pipe.v.position]),
                ),
                pipe.kind,
                has_spatial_up_or_down_pipe_in_timeslice=(
                    pipe.kind.is_spatial and pipe.u.position.z in extended
                ),
                at_temporal_hadamard_layer=at_hadamard,
            ),
            _DEFAULT_BLOCK_REPETITIONS,
        )

    spatial: dict[Pipe, Block] = {}
    for pipe in working.pipes:
        if pipe.kind.is_temporal or component_at[pipe.u.position] in coarse:
            continue
        try:
            spatial[pipe] = build_pipe(pipe)
        except NotImplementedError as exc:
            coarse[component_at[pipe.u.position]].append(
                f"Spatial pipe {pipe.u.position} → {pipe.v.position} is unsupported "
                f"by {convention.name}: {str(exc) or 'no physical pipe implementation'}"
            )

    for component, reasons in coarse.items():
        result.coarse_regions.append(CoarseRegion(component, None, None, " ".join(reasons)))

    # Classify all temporal links before applying substitutions. A boundary mask
    # covers its spatial surgery component, but never spreads to another time slice.
    masks: dict[tuple[tuple[Position3D, ...], int], list[str]] = defaultdict(list)
    temporal: dict[Pipe, Block] = {}
    hadamard_slices: set[int] = set()
    temporal_pipes = sorted(
        (p for p in working.pipes if p.kind.is_temporal),
        key=lambda p: (not p.kind.has_hadamard, p.u.position, p.v.position),
    )
    for pipe in temporal_pipes:
        source, target = component_at[pipe.u.position], component_at[pipe.v.position]
        reason: str | None = None
        if source in coarse or target in coarse:
            reason = "Temporal connection touches a region without a physical implementation."
        else:
            try:
                temporal[pipe] = build_pipe(pipe, pipe.u.position.z in hadamard_slices)
                if pipe.kind.has_hadamard:
                    hadamard_slices.add(pipe.u.position.z)
            except NotImplementedError as exc:
                reason = (
                    f"Temporal pipe is unsupported by {convention.name}: "
                    f"{str(exc) or 'no physical pipe implementation'}"
                )
        if reason is not None:
            result.coarse_regions.append(
                CoarseRegion((), (pipe.u.position, pipe.v.position), None, reason)
            )
            for component, boundary in ((source, 2 * k), (target, 0)):
                if component not in coarse:
                    masks[component, boundary].append(reason)
    for (component, boundary), reasons in masks.items():
        result.coarse_regions.append(
            CoarseRegion(component, None, boundary, " ".join(dict.fromkeys(reasons)))
        )

    compiled = TopologicalComputationGraph(
        _DEFAULT_SCALABLE_QUBIT_SHAPE,
        observable_builder=convention.triplet.observable_builder,
    )
    for pos, block in cubes.items():
        if component_at[pos] not in coarse:
            compiled.add_cube(_position(pos), block)
    # These operations share the strict compiler's temporal substitutions and
    # spatial border trimming. Temporal substitutions must precede spatial pipes.
    for pipe, block in temporal.items():
        compiled.add_pipe(_position(pipe.u.position), _position(pipe.v.position), block)
    for pipe, block in spatial.items():
        if component_at[pipe.u.position] not in coarse:
            compiled.add_pipe(_position(pipe.u.position), _position(pipe.v.position), block)

    cache: dict[tuple[tuple[LayoutPosition2D, int], ...], tuple[LayoutLayer, ScheduledCircuit]] = {}
    fragment_ranges: dict[tuple[Position3D, ...], tuple[int, int, int]] = {}
    next_fragment = 0

    def add_rounds(
        component: tuple[Position3D, ...],
        blocks: dict[LayoutPosition2D, Block],
        transition: bool = False,
    ) -> None:
        nonlocal next_fragment
        expanded = {pos: tuple(block.all_layers(k)) for pos, block in blocks.items()}
        durations = {len(layers) for layers in expanded.values()}
        if len(durations) != 1:
            raise TQECError("Parallel physical blocks have inconsistent round counts.")
        fragment, ticks, measurements = fragment_ranges.get(component, (-1, 0, 0))
        for local_index in range(next(iter(durations))):
            round_index = local_index + (2 * k + 1 if transition else 0)
            if (component, round_index) in masks:
                fragment = -1
                ticks = measurements = 0
                continue
            if fragment < 0:
                fragment = next_fragment
                next_fragment += 1
            layers = {pos: layers[local_index] for pos, layers in expanded.items()}
            cache_key = tuple((pos, id(layer)) for pos, layer in layers.items())
            if cache_key not in cache:
                layer = LayoutLayer(layers, _DEFAULT_SCALABLE_QUBIT_SHAPE)
                cache[cache_key] = layer, layer.to_circuit(k)
            layer, circuit = cache[cache_key]
            raw = circuit.get_circuit()
            result.cycles.append(
                PatchCycle(
                    index=0,
                    block_z=component[0].z,
                    round_index=round_index,
                    component=component,
                    layer=layer,
                    circuit=circuit,
                    is_transition=transition,
                    tick_start=ticks,
                    tick_end=ticks + raw.num_ticks + 1,
                    measurement_start=measurements,
                    measurement_end=measurements + raw.num_measurements,
                    fragment_id=fragment,
                )
            )
            ticks += raw.num_ticks + 1
            measurements += raw.num_measurements
        fragment_ranges[component] = fragment, ticks, measurements

    for component in components:
        if component in coarse:
            continue
        blocks: dict[LayoutPosition2D, Block] = {
            LayoutPosition2D.from_block_position(_position(pos).as_2d()): compiled.get_cube(
                _position(pos)
            )
            for pos in component
        }
        for pipe, block in spatial.items():
            if pipe.u.position in component:
                blocks[
                    LayoutPosition2D.from_pipe_position(
                        (_position(pipe.u.position).as_2d(), _position(pipe.v.position).as_2d())
                    )
                ] = block
        add_rounds(component, blocks)
        transitions: dict[LayoutPosition2D, Block] = {}
        transition_sources: list[Position3D] = []
        for pipe, block in temporal.items():
            if pipe.u.position not in component or len(block.layer_sequence) <= 2:
                continue
            pos = LayoutPosition2D.from_block_position(_position(pipe.u.position).as_2d())
            transitions[pos] = Block(block.layer_sequence[1:-1])
            transition_sources.append(pipe.u.position)
        if transitions:
            add_rounds(tuple(sorted(transition_sources)), transitions, transition=True)

    result.cycles.sort(key=lambda cycle: (cycle.block_z, cycle.round_index, cycle.component))
    _align_parallel_measurements(result.cycles, k)
    known_rounds = sorted({(cycle.block_z, cycle.round_index) for cycle in result.cycles})
    round_indices = {key: i for i, key in enumerate(known_rounds)}
    offsets: dict[int, tuple[int, int]] = {}
    for index, cycle in enumerate(result.cycles):
        cycle.index = index
        # Rescheduling a parallel component can change its tick count. Recompute
        # ranges from the final circuits, preserving their explicit fragment scope.
        cycle.tick_start, cycle.measurement_start = offsets.get(cycle.fragment_id, (0, 0))
        raw = cycle.circuit.get_circuit()
        cycle.tick_end = cycle.tick_start + raw.num_ticks + 1
        cycle.measurement_end = cycle.measurement_start + raw.num_measurements
        offsets[cycle.fragment_id] = cycle.tick_end, cycle.measurement_end
        if result.fully_supported:
            cycle.global_round_index = round_indices[cycle.block_z, cycle.round_index]
    if result.fully_supported:
        result.compiled_graph = compiled
    else:
        result.notices.append(
            "Coarse regions have unknown physical duration; round numbers are local to each block."
        )
    return result
