"""Assemble circuit-backed patch geometry and local topological fallbacks."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tqec.compile.convention import Convention
from tqec.computation._correlation import _partition_graph_into_connected_components
from tqec.computation.block_graph import BlockGraph, BlockKind
from tqec.computation.correlation import CorrelationSurface, find_correlation_surfaces
from tqec.computation.cube import ConditionalCubeKind, Cube
from tqec.computation.pipe import Pipe, PipeKind
from tqec.interop.collada._correlation import CorrelationSurfaceTransformationHelper
from tqec.interop.collada._geometry import BlockGeometries
from tqec.interop.collada.read_write import _Transformation
from tqec.interop.pyzx.positioned import PositionedZX
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Position3D, SignedDirection3D
from tqec.visualisation.patches.analysis import analyse_computation
from tqec.visualisation.patches.boundaries import mark_trace_boundaries
from tqec.visualisation.patches.lowering import lower_block_graph
from tqec.visualisation.patches.model import PatchComputation


@dataclass(frozen=True)
class _SceneCoordinates:
    distance: int
    minimum_z: int
    gaps: frozenset[int]

    @property
    def stride(self) -> int:
        return 2 * self.distance + 2

    def start(self, z: int) -> float:
        return (z - self.minimum_z) * self.distance + sum(g < z for g in self.gaps)

    def cube_box(self, position: Position3D) -> tuple[list[float], list[float]]:
        return (
            [self.stride * position.x, self.stride * position.y, self.start(position.z)],
            [2 * self.distance, 2 * self.distance, self.distance],
        )

    def map_vertex(self, vertex: Iterable[float]) -> list[float]:
        """Map the existing unit-cube/one-unit-pipe primitives to physical footprints."""
        result: list[float] = []
        for axis, value in enumerate(vertex):
            block = math.floor((float(value) + 1e-7) / 2)
            local = float(value) - 2 * block
            if axis < 2:
                offset = (
                    2 * self.distance * local if local <= 1 else 2 * self.distance + 2 * (local - 1)
                )
                result.append(block * self.stride + offset)
            else:
                start = self.start(block)
                if local <= 1:
                    result.append(start + self.distance * local)
                else:
                    gap = self.start(block + 1) - start - self.distance
                    result.append(start + self.distance + gap * (local - 1))
        return result


def _select_surfaces(
    graph: BlockGraph,
    observables: list[CorrelationSurface] | Literal["auto"] | None,
    notices: list[str],
) -> list[tuple[str, CorrelationSurface]]:
    if observables is None:
        return []
    if observables == "auto":
        if graph.has_conditional_cubes:
            notices.append(
                "Automatic correlation discovery is unavailable for unresolved conditional "
                "cubes. Stored conditions remain visible; explicit partial surfaces are accepted."
            )
            surfaces = []
        elif graph.cubes:
            positioned = graph.to_zx_graph()
            surfaces = []
            for component in _partition_graph_into_connected_components(positioned.g):
                if component.num_vertices() > 1 and not any(
                    component.vertex_degree(v) == 1 for v in component.vertices()
                ):
                    notices.append(
                        "Automatic correlation discovery is unavailable for a component "
                        "without leaf nodes. Explicit surfaces can still be displayed."
                    )
                    continue
                surfaces.extend(
                    find_correlation_surfaces(
                        PositionedZX(component, {v: positioned[v] for v in component.vertices()})
                    )
                )
            if not surfaces:
                notices.append("No deterministic correlation surfaces were found.")
        else:
            surfaces = []
    elif isinstance(observables, list) and all(
        isinstance(surface, CorrelationSurface) for surface in observables
    ):
        surfaces = observables
    else:
        raise TQECError('observables must be "auto", a list of correlation surfaces, or None.')
    selected = [(f"Observable {i}", surface) for i, surface in enumerate(surfaces)]
    for cube in graph.conditional_cubes:
        if cube.condition is not None and cube.condition not in [
            surface for _, surface in selected
        ]:
            selected.append((f"Condition for {cube.label or cube.position}", cube.condition))
    for _, surface in selected:
        for pos in surface.positions:
            if pos not in graph:
                raise TQECError(f"Correlation surface references missing cube at {pos}.")
        for edge in surface.span:
            if not edge.is_self_loop and not graph.has_pipe_between(
                edge.u.position, edge.v.position
            ):
                raise TQECError(
                    f"Correlation surface references missing pipe between "
                    f"{edge.u.position} and {edge.v.position}."
                )
    return selected


def _geometry_faces(
    geometries: BlockGeometries,
    kind: BlockKind,
    lower: list[float],
    size: list[float],
    popped: frozenset[SignedDirection3D] = frozenset(),
) -> list[dict[str, Any]]:
    faces = geometries.get_geometry(kind, popped)
    scale = np.asarray(size, dtype=float)
    if isinstance(kind, PipeKind):
        scale[kind.direction.value] /= 2
    elif str(kind) == "Y":
        scale[2] *= 2
    if isinstance(kind, ConditionalCubeKind):
        faces = faces + geometries.get_conditional_frame_geometry(kind, popped)
    return [
        {
            "vertices": (face.get_vertices().reshape(-1, 3) * scale + lower).tolist(),
            "color": face.color.rgba.to_hex(),
        }
        for face in faces
    ]


def _coarse_cube(
    cube: Cube,
    round_index: int | None,
    reason: str,
    coordinates: _SceneCoordinates,
    graph: BlockGraph,
    geometries: BlockGeometries,
) -> dict[str, Any]:
    lower, size = coordinates.cube_box(cube.position)
    kind = "slab" if round_index is not None else "cube"
    if round_index is not None:
        lower[2] += round_index
        size[2] = 1
    elif cube.is_y_cube:
        size[2] /= 2
        if graph.has_pipe_between(cube.position, cube.position.shift_by(dz=1)):
            lower[2] += size[2]
    elif cube.is_port:
        kind = "port"
        pipe = graph.pipes_at(cube.position)[0]
        axis = pipe.direction.value
        if pipe.at_head(cube.position):
            lower[axis] += size[axis]
        size[axis] = 0.1
        lower[axis] -= size[axis] / 2
    popped = frozenset(
        SignedDirection3D(pipe.direction, cube == pipe.u)
        for pipe in graph.pipes_at(cube.position)
        if round_index is None or pipe.kind.is_spatial
    )
    return {
        "type": kind,
        "center": [v + s / 2 for v, s in zip(lower, size)],
        "size": size,
        "kind": str(cube.kind),
        "reason": reason,
        "label": f"{cube.label or cube.kind} at {cube.position}"
        + (f", round {round_index + 1}" if round_index is not None else ""),
        "conditional": (
            [str(branch) for branch in cube.kind.branches]
            if isinstance(cube.kind, ConditionalCubeKind)
            else None
        ),
        "faces": []
        if cube.is_port
        else _geometry_faces(geometries, cube.kind, lower, size, popped),
    }


def _coarse_pipe(
    pipe: Pipe,
    round_index: int | None,
    reason: str,
    coordinates: _SceneCoordinates,
    geometries: BlockGeometries,
) -> dict[str, Any]:
    lower, size = coordinates.cube_box(pipe.u.position)
    upper, _ = coordinates.cube_box(pipe.v.position)
    axis = pipe.direction.value
    lower[axis] += size[axis]
    size[axis] = max(0.08, upper[axis] - lower[axis])
    if round_index is not None:
        lower[2] += round_index
        size[2] = 1
    return {
        "type": "pipe",
        "center": [v + s / 2 for v, s in zip(lower, size)],
        "size": size,
        "kind": str(pipe.kind),
        "reason": reason,
        "label": f"Pipe {pipe.u.position} → {pipe.v.position}",
        "faces": _geometry_faces(geometries, pipe.kind, lower, size),
    }


def _coarse_geometry(
    computation: PatchComputation, coordinates: _SceneCoordinates
) -> list[dict[str, Any]]:
    geometries = BlockGeometries()
    items: dict[tuple[Any, ...], dict[str, Any]] = {}
    for region in computation.coarse_regions:
        if region.pipe is not None:
            pipe = computation.graph.get_pipe(*region.pipe)
            items[("pipe", *region.pipe, None)] = _coarse_pipe(
                pipe, None, region.reason, coordinates, geometries
            )
        for pos in region.cubes:
            items["cube", pos, region.round_index] = _coarse_cube(
                computation.graph[pos],
                region.round_index,
                region.reason,
                coordinates,
                computation.graph,
                geometries,
            )
        for pipe in computation.graph.pipes:
            if (
                pipe.kind.is_spatial
                and pipe.u.position in region.cubes
                and pipe.v.position in region.cubes
            ):
                items["pipe", pipe.u.position, pipe.v.position, region.round_index] = _coarse_pipe(
                    pipe, region.round_index, region.reason, coordinates, geometries
                )
    return list(items.values())


def _surface_vertices(
    transformation: _Transformation, coordinates: _SceneCoordinates
) -> list[list[float]]:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    vertices = (
        vertices * transformation.scale
    ) @ transformation.rotation.T + transformation.translation
    return [coordinates.map_vertex(vertex) for vertex in vertices]


def _runs(rounds: Iterable[int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for index in sorted(rounds):
        if result and result[-1][1] == index:
            result[-1] = result[-1][0], index + 1
        else:
            result.append((index, index + 1))
    return result


def _fallback_surface(
    computation: PatchComputation,
    surface: CorrelationSurface,
    physical_cycle_ids: set[int],
    coordinates: _SceneCoordinates,
) -> list[dict[str, Any]]:
    helper = CorrelationSurfaceTransformationHelper(computation.graph, pipe_length=1)
    covered = {
        (pos, cycle.round_index)
        for cycle in computation.cycles
        if cycle.index in physical_cycle_ids and not cycle.is_transition
        for pos in cycle.component
    }
    coarse_pipes = {region.pipe for region in computation.coarse_regions if region.pipe is not None}
    items: list[dict[str, Any]] = []

    def append_piece(
        basis: Basis,
        transform: _Transformation,
        z: int,
        interval: tuple[int, int] | None = None,
        fixed_z: float | None = None,
    ) -> None:
        vertices = _surface_vertices(transform, coordinates)
        if interval is not None:
            start, end = interval
            base = coordinates.start(z)
            for vertex in vertices:
                vertex[2] = base + start + (vertex[2] - base) / coordinates.distance * (end - start)
        if fixed_z is not None:
            for vertex in vertices:
                vertex[2] = fixed_z
        items.append(
            {
                "vertices": vertices,
                "basis": basis.value,
                "reason": "Topological correlation surface: physical tracking is unavailable here.",
            }
        )

    for pos in sorted(surface.positions):
        missing = [r for r in range(coordinates.distance) if (pos, r) not in covered]
        pieces = helper._compute_partial_cube_transformations(pos, surface.edges_at(pos))
        if computation.graph[pos].is_y_cube:
            # The partial-surface helper uses COLLADA's overlapping Y endpoint
            # placement. Here half-cubes stay inside their block-time footprint.
            # Its only Y interior pieces are singleton tiles, which belong at
            # the center of the actual half-cube rather than in the adjacent pipe.
            fraction = 0.75 if computation.graph.has_pipe_between(pos, pos.shift_by(dz=1)) else 0.25
            for basis, transform in pieces:
                height = coordinates.start(pos.z) + coordinates.distance * (
                    fraction + (-0.01 if basis == Basis.X else 0.01)
                )
                append_piece(basis, transform, pos.z, fixed_z=height)
            continue
        for interval in _runs(missing):
            for basis, transform in pieces:
                append_piece(basis, transform, pos.z, interval)
    for edge in sorted(surface.span):
        if edge.is_self_loop:
            continue
        pipe = computation.graph.get_pipe(edge.u.position, edge.v.position)
        pieces = helper._compute_pipe_transformations(edge)
        if pipe.kind.is_spatial:
            missing = [
                r
                for r in range(coordinates.distance)
                if (pipe.u.position, r) not in covered or (pipe.v.position, r) not in covered
            ]
            for interval in _runs(missing):
                for basis, transform in pieces:
                    append_piece(basis, transform, pipe.u.position.z, interval)
        elif (
            (pipe.u.position, pipe.v.position) in coarse_pipes
            or (pipe.u.position, coordinates.distance - 1) not in covered
            or (pipe.v.position, 0) not in covered
            or any(
                cycle.is_transition
                and pipe.u.position in cycle.component
                and cycle.index not in physical_cycle_ids
                for cycle in computation.cycles
            )
        ):
            for basis, transform in pieces:
                append_piece(basis, transform, pipe.u.position.z)
    return items


def _boundary_glyphs(cycle: Any, qubits: list[dict[str, Any]]) -> None:
    lookup = {(q["x"], q["y"]): q for q in qubits if q["role"] == "data"}
    for instruction in cycle.circuit.get_circuit(include_qubit_coords=False):
        name = instruction.name
        if name not in {"R", "RX", "RY", "M", "MX", "MY", "MR", "MRX", "MRY"}:
            continue
        for target in instruction.targets_copy():
            qubit = cycle.circuit.qubit_map.i2q[target.value]
            record = lookup.get((qubit.x, qubit.y))
            if record is None:
                continue
            basis = "X" if name.endswith("X") else "Y" if name.endswith("Y") else "Z"
            if name.startswith("M"):
                record["measurement"] = basis
            if name.startswith("R") or name.startswith("MR"):
                record["reset"] = basis


def build_patch_scene(
    graph: BlockGraph,
    k: int,
    convention: Convention,
    observables: list[CorrelationSurface] | Literal["auto"] | None = "auto",
) -> dict[str, Any]:
    """Build a JSON-compatible scene without modifying the graph or any detector database."""
    computation = lower_block_graph(graph, k, convention)
    notices = list(computation.notices)
    selected = _select_surfaces(graph, observables, notices)
    analysis = analyse_computation(computation, [surface for _, surface in selected])
    notices.extend(analysis.notices)
    gaps = {cycle.block_z for cycle in computation.cycles if cycle.is_transition}
    gaps.update(
        region.pipe[0].z for region in computation.coarse_regions if region.pipe is not None
    )
    coordinates = _SceneCoordinates(
        computation.distance,
        min((cube.position.z for cube in graph.cubes), default=0),
        frozenset(gaps),
    )
    cycles = []
    for cycle in computation.cycles:
        record = analysis.cycles[cycle.index].to_dict()
        _boundary_glyphs(cycle, record["qubits"])
        record.update(
            {
                "id": cycle.index,
                "block_z": cycle.block_z,
                "round": cycle.round_index,
                "global_round_index": cycle.global_round_index,
                "time": coordinates.start(cycle.block_z) + cycle.round_index + 0.5,
                "label": f"Block z={cycle.block_z} · "
                + (
                    "Hadamard realignment"
                    if cycle.is_transition
                    else f"round {cycle.round_index + 1}/{computation.distance}"
                ),
                "component": [list(pos.as_tuple()) for pos in cycle.component],
                "fragment_id": cycle.fragment_id,
                "tick_range": [cycle.tick_start, cycle.tick_end],
                "measurement_range": [cycle.measurement_start, cycle.measurement_end],
            }
        )
        cycles.append(record)
    outlines = []
    for cube in graph.cubes:
        lower, size = coordinates.cube_box(cube.position)
        outlines.append(
            {
                "center": [v + s / 2 for v, s in zip(lower, size)],
                "size": size,
                "label": f"{cube.label or cube.kind} at {cube.position}",
            }
        )
    overlay_records = []
    for (label, surface), overlay in zip(selected, analysis.observables):
        record = dict(overlay)
        record["name"] = label
        record["cycles"] = mark_trace_boundaries(computation, surface, record.get("cycles", []))
        record["surfaces"] = _fallback_surface(
            computation,
            surface,
            {item["cycle_id"] for item in record.get("cycles", [])},
            coordinates,
        )
        overlay_records.append(record)
    if computation.coarse_regions:
        notices.append(
            "Blocks and boundary slabs mark unavailable physical operations. "
            "Global cycle offsets across these regions are unspecified."
        )
    return {
        "name": graph.name or "Surface-code patches",
        "k": k,
        "d": computation.distance,
        "cycles": cycles,
        "coarse": _coarse_geometry(computation, coordinates),
        "outlines": outlines,
        "observables": overlay_records,
        "notices": list(dict.fromkeys(notices)),
    }
