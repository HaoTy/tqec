"""Identify where established physical logical traces meet unavailable intervals."""

from __future__ import annotations

from typing import Any

from tqec.computation.correlation import CorrelationSurface
from tqec.utils.position import Position3D
from tqec.visualisation.patches.model import PatchComputation, PatchCycle


def mark_trace_boundaries(
    computation: PatchComputation,
    surface: CorrelationSurface,
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate trace endpoints only where their surface continues into an unknown region.

    Ordinary preparation and readout caps are not tracking losses. Empty but
    established physical snapshots also remain part of a trace. Boundary
    detection uses circuit coverage and surface connectivity rather than the
    display coordinates of topological polygons.

    """
    by_id = {cycle.index: cycle for cycle in computation.cycles}
    slots = {
        (pos, cycle.round_index): cycle for cycle in computation.cycles for pos in cycle.component
    }
    tracked = {snapshot["cycle_id"] for snapshot in snapshots}
    temporal: dict[tuple[Position3D, int], Position3D] = {}
    for edge in surface.span:
        source, target = edge.u.position, edge.v.position
        if source.z == target.z:
            continue
        direction = 1 if target.z > source.z else -1
        temporal[source, direction] = target
        temporal[target, -direction] = source
    coarse_pipes = {
        region.pipe: region.reason
        for region in computation.coarse_regions
        if region.pipe is not None
    }

    def unavailable_reason(pos: Position3D, round_index: int) -> str:
        for region in computation.coarse_regions:
            if pos in region.cubes and region.round_index in (None, round_index):
                return region.reason
        return "The adjacent logical operator could not be verified at the physical level."

    def adjacent_reason(cycle: PatchCycle, pos: Position3D, direction: int) -> str | None:
        adjacent_index = cycle.round_index + direction
        peer_pos = pos
        if adjacent_index < 0:
            peer_pos = temporal.get((pos, -1))
            if peer_pos is None:
                return None
            pipe = (peer_pos, pos)
            if pipe in coarse_pipes:
                return coarse_pipes[pipe]
            adjacent_index = max(
                computation.distance - 1,
                max((index for position, index in slots if position == peer_pos), default=-1),
            )
        elif adjacent_index >= computation.distance and (pos, adjacent_index) not in slots:
            peer_pos = temporal.get((pos, 1))
            if peer_pos is None:
                return None
            pipe = (pos, peer_pos)
            if pipe in coarse_pipes:
                return coarse_pipes[pipe]
            adjacent_index = 0
        peer = slots.get((peer_pos, adjacent_index))
        if peer is not None and peer.index in tracked:
            return None
        return unavailable_reason(peer_pos, adjacent_index)

    result = []
    for snapshot in snapshots:
        record = dict(snapshot)
        cycle = by_id[snapshot["cycle_id"]]
        boundaries = []
        if snapshot.get("support"):
            for side, direction in (("past", -1), ("future", 1)):
                reasons = [
                    reason
                    for pos in cycle.component
                    if pos in surface.positions
                    if (reason := adjacent_reason(cycle, pos, direction)) is not None
                ]
                if reasons:
                    boundaries.append({"side": side, "reason": " ".join(dict.fromkeys(reasons))})
        if boundaries:
            record["boundaries"] = boundaries
        result.append(record)
    return result
