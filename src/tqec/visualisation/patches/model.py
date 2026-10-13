"""Physical rounds and explicitly unavailable regions of a hybrid computation."""

from __future__ import annotations

from dataclasses import dataclass, field

from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.convention import Convention
from tqec.compile.graph import TopologicalComputationGraph
from tqec.computation.block_graph import BlockGraph
from tqec.utils.position import Position3D


@dataclass
class PatchCycle:
    """One compiled round in a connected spatial component.

    Tick and measurement ranges are half-open, relative to the contiguous
    component fragment identified by ``fragment_id``. A new block slice or
    masked interval starts a new fragment. ``index`` identifies the scene record,
    not elapsed time.
    ``global_round_index`` is absent when physical durations are unavailable.
    """

    index: int
    block_z: int
    round_index: int
    component: tuple[Position3D, ...]
    layer: LayoutLayer
    circuit: ScheduledCircuit
    is_transition: bool = False
    tick_start: int = 0
    tick_end: int = 0
    measurement_start: int = 0
    measurement_end: int = 0
    global_round_index: int | None = None
    fragment_id: int = 0


@dataclass(frozen=True)
class CoarseRegion:
    """A whole spatial component, temporal pipe, or unavailable boundary round."""

    cubes: tuple[Position3D, ...]
    pipe: tuple[Position3D, Position3D] | None
    round_index: int | None
    reason: str


@dataclass
class PatchComputation:
    """A compiled scene retaining the source graph and its coordinate system."""

    graph: BlockGraph
    convention: Convention
    k: int
    cycles: list[PatchCycle] = field(default_factory=list)
    coarse_regions: list[CoarseRegion] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    compiled_graph: TopologicalComputationGraph | None = None

    @property
    def fully_supported(self) -> bool:
        """Whether every region has a known physical implementation."""
        return not self.coarse_regions

    @property
    def distance(self) -> int:
        """Surface-code distance using the compiler's scaling convention."""
        return 2 * self.k + 1
