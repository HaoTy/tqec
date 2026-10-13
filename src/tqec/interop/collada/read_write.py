"""Read and write block graphs to and from Collada DAE files.

Y half-cubes: integer positions in BlockGraph/BGRAPH; ±0.5 Z offset in DAE for visual rendering.
Writer applies the offset via offset_y_half_cube_position; readers recover via
int_position_before_scale (atol=0.35 absorbs the 0.5/(1+pipe_length) residual for all
pipe_length values usable in a 3D GUI).
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import BinaryIO, cast

import collada
import collada.source
import numpy as np
import numpy.typing as npt

from tqec.computation.block_graph import BlockGraph, BlockKind, block_kind_from_str
from tqec.computation.correlation import CorrelationSurface
from tqec.computation.cube import ConditionalCubeKind, CubeKind, LeafCubeKind
from tqec.computation.pipe import PipeKind
from tqec.interop.collada._geometry import (
    BlockGeometries,
    Face,
    get_correlation_surface_geometry,
)
from tqec.interop.color import TQECColor
from tqec.interop.shared import (
    int_position_before_scale,
    offset_y_half_cube_position,
    scale_position,
)
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Direction3D, FloatPosition3D, Position3D, SignedDirection3D
from tqec.utils.rotations import (
    adjust_hadamards_direction,
    get_axes_directions,
    rotate_on_import,
)

_ASSET_AUTHOR = "TQEC Community"
_ASSET_AUTHORING_TOOL_TQEC = "https://github.com/tqec/tqec"
_ASSET_UNIT_NAME = "inch"
_ASSET_UNIT_METER = 0.02539999969303608

_MATERIAL_SYMBOL = "MaterialSymbol"
_CORRELATION_SUFFIX = "_CORRELATION"
_CONDITION_LINK_SUFFIX = "_CONDITION_LINK"
_CONDITIONAL_FRAME_SUFFIX = "_CONDITIONAL_FRAME"


# DAE EXPORTER/IMPORTER
def read_block_graph_from_dae_file(
    filepath: str | pathlib.Path,
    graph_name: str = "",
) -> BlockGraph:
    """Read a Collada DAE file and construct a :class:`.BlockGraph` from it.

    Args:
        filepath: The input dae file path.
        graph_name: The name of the block graph. Default is an empty string.

    Returns:
        The constructed :py:class:`~tqec.computation.block_graph.BlockGraph` object.

    Raises:
        TQECError: If the COLLADA model cannot be parsed and converted to a block graph.

    """
    # Bring the mesh in
    mesh = collada.Collada(str(filepath))

    # Check some invariants about the DAE file
    if mesh.scene is None:
        raise TQECError("No scene found in the DAE file.")
    scene: collada.scene.Scene = mesh.scene

    if not (len(scene.nodes) == 1 and scene.nodes[0].name == "SketchUp"):
        raise TQECError(
            "The <visual_scene> node must have a single child node with the name 'SketchUp'."
        )

    sketchup_node: collada.scene.Node = scene.nodes[0]
    pipe_length: float | None = None
    parsed_cubes: list[tuple[FloatPosition3D, CubeKind, dict[str, int]]] = []
    parsed_pipes: list[tuple[FloatPosition3D, PipeKind, dict[str, int]]] = []

    # Handle nodes in scene/sketchup_node
    for node in sketchup_node.children:
        # If everything needed is present
        if (
            isinstance(node, collada.scene.Node)
            and node.matrix is not None
            and node.children is not None
            and len(node.children) == 1
            and isinstance(node.children[0], collada.scene.NodeNode)
        ):
            # Extract key info from collada scene
            instance = cast(collada.scene.NodeNode, node.children[0])
            library_node: collada.scene.Node = instance.node
            name: str = library_node.name

            # Skip visualization annotations.
            if name.endswith(
                (_CORRELATION_SUFFIX, _CONDITION_LINK_SUFFIX, _CONDITIONAL_FRAME_SUFFIX)
            ):
                continue

            # Extract transformation and translation matrix
            transformation = _Transformation.from_4d_affine_matrix(node.matrix)
            translation = FloatPosition3D(*transformation.translation)
            axes_directions = get_axes_directions(transformation.rotation)
            kind = block_kind_from_str(name)
            if isinstance(kind, ConditionalCubeKind):
                raise TQECError(
                    "Cannot import conditional cubes from a visualization-only DAE file: "
                    "their conditions cannot be reconstructed. Use JSON for round trips."
                )

            # Rotations step 1. Skip if node's matrix not rotated
            # - If node's matrix YES rotated: check closer & make necessary adjustments
            if not np.allclose(transformation.rotation, np.eye(3), atol=1e-9):
                translation, kind = rotate_on_import(
                    transformation.rotation,
                    transformation.translation,
                    transformation.scale,
                    kind,
                )

            # Rotations step 2. Skip if hadamard points in positive direction
            if isinstance(kind, PipeKind):
                if axes_directions[str(kind.direction)] == -1 and "H" in str(kind):
                    kind = adjust_hadamards_direction(kind)

            # Direction, scaling and checks for pipes
            if isinstance(kind, PipeKind):
                # Get direction and scale of pipe
                pipe_direction = kind.direction
                scale = transformation.scale[pipe_direction.value]
                # Checks
                if pipe_length is None:
                    pipe_length = scale * 2.0
                elif not np.isclose(pipe_length, scale * 2.0, atol=1e-9):
                    raise TQECError("All pipes must have the same length.")
                expected_scale = np.ones(3)
                expected_scale[pipe_direction.value] = scale
                if not np.allclose(transformation.scale, expected_scale, atol=1e-9):
                    raise TQECError(
                        "Only the dimension along the pipe can be scaled, "
                        f"which is not the case at {translation}."
                    )
                # Append
                parsed_pipes.append((translation, kind, axes_directions))

            else:
                # Checks
                if not np.allclose(transformation.scale, np.ones(3), atol=1e-9):
                    raise TQECError(f"Cube at {translation} has a non-identity scale.")
                # Append
                parsed_cubes.append((translation, kind, axes_directions))

    pipe_length = 2.0 if pipe_length is None else pipe_length

    # Construct graph
    # Create graph
    graph = BlockGraph(graph_name)

    # Add cubes (int_position_before_scale absorbs the ±0.5 DAE visual offset for Y half-cubes)
    for pos, cube_kind, axes_directions in parsed_cubes:
        graph.add_cube(int_position_before_scale(pos, pipe_length), cube_kind)
    port_index = 0

    # Add pipes
    for pos, pipe_kind, axes_directions in parsed_pipes:
        # Draw pipes in +1/-1 direction using position, kind of pipe, and directional pointers
        # from previous operations
        directional_multiplier = axes_directions[str(pipe_kind.direction)]
        head_pos = int_position_before_scale(
            pos.shift_in_direction(pipe_kind.direction, -1 * directional_multiplier),
            pipe_length,
        )
        tail_pos = head_pos.shift_in_direction(pipe_kind.direction, 1 * directional_multiplier)

        # Add pipe
        if head_pos not in graph:
            graph.add_cube(head_pos, LeafCubeKind.PORT, label=f"Port{port_index}")
            port_index += 1
        if tail_pos not in graph:
            graph.add_cube(tail_pos, LeafCubeKind.PORT, label=f"Port{port_index}")
            port_index += 1
        graph.add_pipe(head_pos, tail_pos, pipe_kind)

    return graph


def write_block_graph_to_dae_file(
    block_graph: BlockGraph,
    file_like: str | pathlib.Path | BinaryIO,
    pipe_length: float = 2.0,
    pop_faces_at_directions: Iterable[SignedDirection3D | str] = (),
    show_correlation_surface: CorrelationSurface | None = None,
    *,
    show_conditions: bool = True,
) -> None:
    """Write a :py:class:`~tqec.computation.block_graph.BlockGraph` to a Collada DAE file.

    Args:
        block_graph: The block graph to write to the DAE file.
        file_like: The output file path or file-like object that supports binary write.
        pipe_length: The length of the pipes in the COLLADA model. Default is 2.0.
        pop_faces_at_directions: Remove the faces at the given directions for all the blocks.
            This is useful for visualizing the internal structure of the blocks.
            Default is an empty tuple.
        show_correlation_surface: The :py:class:`~tqec.computation.correlation.CorrelationSurface`
            to show in the block graph. Default is None.
        show_conditions: Show all conditional cubes' controlling surfaces and rounded exterior
            dashed links, making only blocks containing those surfaces translucent. Default is True.
            Conditional cubes must have exactly one compatible pipe even when this is False.

    Note:
        Conditional cubes, including Y branches, use full-size split-color symbols with opaque
        white outer frames. Frames remain visible with ``show_conditions=False``.
        DAE export preserves their appearance, not their condition data; use JSON for round trips.

    """
    # Validate conditional leaves even when annotations are disabled. Do not require unrelated
    # static blocks to be valid: the exporter also supports inspecting unfinished graphs.
    conditional_cubes = block_graph.conditional_cubes
    for cube in conditional_cubes:
        block_graph._validate_locally_at_cube(cube)

    # Import lazily to avoid the transformation helper's dependency on this module.
    from tqec.interop.collada._correlation import (  # noqa: PLC0415
        CorrelationSurfaceTransformationHelper,
    )

    helper = CorrelationSurfaceTransformationHelper(block_graph, pipe_length)
    conditions = {cube.condition for cube in conditional_cubes}
    condition_pieces = []
    translucent_cubes: set[Position3D] = set()
    translucent_pipes: set[frozenset[Position3D]] = set()
    if show_conditions:
        for cube in conditional_cubes:
            assert cube.condition is not None
            try:
                pieces = helper.get_transformations_for_partial_surface(cube.condition)
            except TQECError as exc:
                raise TQECError(
                    f"Cannot display condition for cube at {cube.position}: {exc}"
                ) from exc
            if pieces:
                condition_pieces.append((cube.position, pieces))
                for edge in cube.condition.span:
                    if not edge.is_self_loop:
                        translucent_pipes.add(frozenset((edge.u.position, edge.v.position)))
                    for position in (edge.u.position, edge.v.position):
                        endpoint = block_graph[position]
                        if not endpoint.is_port and (edge.is_self_loop or not endpoint.is_y_cube):
                            translucent_cubes.add(position)

    directions: list[SignedDirection3D] = []
    for direction in pop_faces_at_directions:
        if isinstance(direction, str):
            directions.append(SignedDirection3D.from_string(direction))
        else:
            directions.append(direction)
    base = _BaseColladaData(directions)
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)

    for cube in block_graph.cubes:
        if cube.is_port:
            continue
        scaled_position = scale_position(cube.position, pipe_length=pipe_length)
        if cube.is_y_cube:
            if block_graph.has_pipe_between(cube.position, cube.position.shift_by(dz=1)):
                scaled_position = offset_y_half_cube_position(scaled_position, 1)  # init: +0.5
            elif block_graph.has_pipe_between(cube.position, cube.position.shift_by(dz=-1)):
                scaled_position = offset_y_half_cube_position(scaled_position, -1)  # meas: -0.5

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = scaled_position.as_array()
        size = np.array([1.0, 1.0, 0.5 if cube.is_y_cube else 1.0])
        bounds_min = np.minimum(bounds_min, matrix[:3, 3])
        bounds_max = np.maximum(bounds_max, matrix[:3, 3] + size)
        pop_directions = [
            SignedDirection3D(pipe.direction, cube == pipe.u)
            for pipe in block_graph.pipes_at(cube.position)
        ]
        base.add_block_instance(
            matrix, cube.kind, pop_directions, translucent=cube.position in translucent_cubes
        )
        if isinstance(cube.kind, ConditionalCubeKind):
            base.add_conditional_frame(matrix, cube.kind, pop_directions)

    for pipe in block_graph.pipes:
        head_pos = scale_position(pipe.u.position, pipe_length=pipe_length)
        pipe_pos = head_pos.shift_in_direction(pipe.direction, 1.0)

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = pipe_pos.as_array()
        scales = [1.0, 1.0, 1.0]

        # We divide the scaling by 2.0 because the pipe's default length is 2.0.
        scales[pipe.direction.value] = pipe_length / 2.0
        matrix[:3, :3] = np.diag(scales)
        size = np.ones(3)
        size[pipe.direction.value] = pipe_length
        bounds_min = np.minimum(bounds_min, matrix[:3, 3])
        bounds_max = np.maximum(bounds_max, matrix[:3, 3] + size)

        base.add_block_instance(
            matrix,
            pipe.kind,
            translucent=frozenset((pipe.u.position, pipe.v.position)) in translucent_pipes,
        )

    for position, pieces in condition_pieces:
        for basis, transformation in pieces:
            base.add_surface_instance(basis, transformation)
        target = scale_position(position, pipe_length).as_array() + 0.5
        (pipe,) = block_graph.pipes_at(position)
        free_direction = SignedDirection3D(pipe.direction, not pipe.at_head(position))
        base.add_condition_link(
            [transformation.to_4d_affine_matrix() for _, transformation in pieces],
            target,
            free_direction,
            np.array([bounds_min, bounds_max]),
        )

    if show_correlation_surface is not None:
        pieces = (
            helper.get_transformations_for_partial_surface(show_correlation_surface)
            if show_correlation_surface in conditions
            else helper.get_transformations_for_correlation_surface(show_correlation_surface)
        )
        for basis, transformation in pieces:
            base.add_surface_instance(basis, transformation)

    base.mesh.write(file_like)


# JSON IMPORTER USING WEBGL-LIKE COORDINATES
# Potentially useful for other WEBGL tools besides SketchUp
# Why no JSON exporter? No need. It can be done with blockgraph.to_dict
def read_block_graph_from_json(
    filepath: str | pathlib.Path,
    graph_name: str = "",
) -> BlockGraph:
    """Construct a :py:class:`.BlockGraph` from a JSON file with COLLADA-like information.

    Args:
        filepath: The input JSON file path.
        graph_name: The name of the block graph. Default is an empty string.

    Returns:
        The constructed :py:class:`.BlockGraph` object.

    Raises:
        TQECError: If the JSON file cannot be parsed and converted to a block graph.

    """
    # Read JSON file
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception:
        raise TQECError("JSON file not found.")

    # Check JSON file has cubes and pipes
    try:
        cubes = data["cubes"]
        pipes = data["pipes"]
        if not len(cubes) > 0 or not len(pipes) > 0:
            raise TQECError("No cubes or pipes found.")
    except Exception:
        raise TQECError("JSON file is not appropriately formatted.")

    # Initialise list of cubes and pipes
    parsed_cubes: list[
        tuple[FloatPosition3D, CubeKind, dict[str, int], CorrelationSurface | None]
    ] = []
    parsed_pipes: list[tuple[FloatPosition3D, FloatPosition3D, PipeKind, dict[str, int]]] = []

    # Get cubes data
    for cube in data["cubes"]:
        # Skip any "PORT" kind: this JSON schema carries no label, so ports cannot be
        # constructed here. They are instead re-created (with generated labels) from the
        # dangling pipe ends when adding pipes below.
        if cube["kind"] == "PORT":
            continue

        # Enforce integers for position and transformation
        if not all([isinstance(i, int) for i in cube["position"]]):
            raise TQECError(
                f"Incorrect positioning for cube at {cube['position']}. All positional "
                "information must be composed of integer values."
            )

        if not all([isinstance(i, int) for row in cube["transform"] for i in row]):
            raise TQECError(
                f"Incorrect transformation matrix for cube at {cube['position']}. All "
                "elements in transformation matrix need to be integers."
            )

        # Get key spatial info
        translation = FloatPosition3D(*cube["position"])
        axes_directions = get_axes_directions(cube["transform"])
        kind = block_kind_from_str(cube["kind"])

        # Rotations step 1. Skip if node's matrix not rotated
        # - If node's matrix YES rotated: check closer & make necessary adjustments
        if not np.allclose(cube["transform"], np.eye(3), atol=1e-9):
            translation, kind = rotate_on_import(
                cube["transform"],
                cube["position"],
                np.array([1.0, 1.0, 1.0]),
                kind,
                # potential TODO: rotate condition
            )

        # Append to parsed cubes
        if isinstance(kind, CubeKind):
            parsed_cubes.append(
                (
                    translation,
                    kind,
                    axes_directions,
                    None
                    if (condition := cube.get("condition", None)) is None
                    else CorrelationSurface.from_dict(condition),
                )
            )

    # Get pipes data
    for pipe in data["pipes"]:
        # Enforce integers for position and transformation
        for pos in [pipe["u"], pipe["v"]]:
            if not all([isinstance(i, int) for i in pos]):
                raise TQECError(
                    f"Incorrect positioning for pipe at ({pipe['u']}). All positional "
                    "information must be composed of integer values."
                )

        if not all([isinstance(i, int) for row in pipe["transform"] for i in row]):
            raise TQECError(
                f"Incorrect transformation for pipe at ({pipe['u']}). All elements in "
                "transformation matrix need to be integers."
            )

        # Get key spatial info
        u_pos = FloatPosition3D(*pipe["u"])  # Equivalent to "translation" in cubes
        v_pos = FloatPosition3D(*pipe["v"])
        axes_directions = get_axes_directions(pipe["transform"])
        kind = block_kind_from_str(pipe["kind"])

        # Rotations step 1. Skip if node's matrix not rotated
        # - If node's matrix YES rotated: check closer & make necessary adjustments
        if not np.allclose(pipe["transform"], np.eye(3), atol=1e-9):
            u_pos, kind = rotate_on_import(
                pipe["transform"],
                pipe["position"],
                np.array([1.0, 1.0, 1.0]),
                kind,
            )

        # Rotations step 2. Skip if hadamard points in positive direction
        # Check kind is pipe
        if isinstance(kind, PipeKind):
            if axes_directions[str(kind.direction)] == -1 and "H" in str(kind):
                kind = adjust_hadamards_direction(kind)

        # Recheck kind since it might have been regenerated
        if isinstance(kind, PipeKind):
            parsed_pipes.append((u_pos, v_pos, kind, axes_directions))

    # Construct graph
    # Create graph
    graph = BlockGraph(graph_name)

    # JSON positions are exact integers; preserve the conditional expression.
    for pos, cube_kind, axes_directions, condition in parsed_cubes:
        graph.add_cube(int_position_before_scale(pos, 0.0), cube_kind, condition=condition)
    port_index = 0

    # Add pipes
    for u_pos, v_pos, pipe_kind, axes_directions in parsed_pipes:
        # Write head_pos and tail_pos as Position3D
        head_pos = int_position_before_scale(u_pos, 0)
        tail_pos = int_position_before_scale(v_pos, 0)

        # Add pipe
        if head_pos not in graph:
            graph.add_cube(head_pos, LeafCubeKind.PORT, label=f"Port{port_index}")
            port_index += 1
        if tail_pos not in graph:
            graph.add_cube(tail_pos, LeafCubeKind.PORT, label=f"Port{port_index}")
            port_index += 1
        graph.add_pipe(head_pos, tail_pos, pipe_kind)

    return graph


# CLASSES
@dataclass(frozen=True)
class _BlockLibraryKey:
    """The key to access the library node in the Collada DAE file."""

    kind: BlockKind
    pop_faces_at_directions: frozenset[SignedDirection3D] = frozenset()
    translucent: bool = False

    def __str__(self) -> str:
        string = f"{self.kind}"
        if self.pop_faces_at_directions:
            string += "-without-"
            string += "-".join(str(d) for d in self.pop_faces_at_directions)
        if self.translucent:
            string += "_translucent"
        return string


class _BaseColladaData:
    def __init__(
        self,
        pop_faces_at_directions: Iterable[SignedDirection3D] = (),
    ) -> None:
        """Encode the base model template.

        This class includes the definition of all the library nodes and the necessary material,
        geometry definitions.
        """
        self.mesh = collada.Collada()
        self.geometries = BlockGeometries()

        self.materials: dict[tuple[TQECColor, bool], collada.material.Material] = {}
        self.geometry_nodes: dict[tuple[Face, bool], collada.scene.GeometryNode] = {}
        self.root_node = collada.scene.Node("SketchUp", name="SketchUp")
        self.block_library: dict[_BlockLibraryKey, collada.scene.Node] = {}
        self._frame_library: dict[_BlockLibraryKey, collada.scene.Node] = {}
        self.surface_library: dict[Basis, collada.scene.Node] = {}
        self._surface_instances: dict[
            tuple[Basis, bytes], tuple[npt.NDArray[np.float64] | None, collada.scene.Node]
        ] = {}
        self._condition_link_node: collada.scene.Node | None = None
        self._pop_faces_at_directions: frozenset[SignedDirection3D] = frozenset(
            pop_faces_at_directions
        )
        self._num_instances: int = 0

        self._create_scene()
        self._add_asset_info()
        self._add_materials()

    def _create_scene(self) -> None:
        scene = collada.scene.Scene("scene", [self.root_node])
        self.mesh.scenes.append(scene)
        self.mesh.scene = scene

    def _add_asset_info(self) -> None:
        if self.mesh.assetInfo is None:
            return
        self.mesh.assetInfo.contributors.append(
            collada.asset.Contributor(
                author=_ASSET_AUTHOR, authoring_tool=_ASSET_AUTHORING_TOOL_TQEC
            ),
        )
        self.mesh.assetInfo.unitmeter = _ASSET_UNIT_METER
        self.mesh.assetInfo.unitname = _ASSET_UNIT_NAME
        self.mesh.assetInfo.upaxis = collada.asset.UP_AXIS.Z_UP

    def _add_materials(self) -> None:
        """Add all the materials for different faces."""
        for face_color in TQECColor:
            self._add_material(face_color)

    def _add_material(self, face_color: TQECColor, translucent: bool = False) -> None:
        key = (face_color, translucent)
        if key in self.materials:
            return
        color = face_color.rgba.with_alpha(0.3) if translucent else face_color.rgba
        rgba = color.as_floats()
        suffix = "_translucent" if translucent else ""
        effect = collada.material.Effect(
            f"{face_color.value}_effect{suffix}",
            [],
            "lambert",
            diffuse=rgba,
            emission=None,
            specular=None,
            # COLLADA A_ONE multiplies transparent alpha by transparency. Use a unit alpha
            # here so loaders apply the requested opacity once, rather than squaring it.
            transparent=(1.0, 1.0, 1.0, 1.0),
            transparency=rgba[3],
            ambient=None,
            reflective=None,
            double_sided=True,
        )
        self.mesh.effects.append(effect)
        material_id = f"{face_color.value}_material{suffix}"
        material = collada.material.Material(material_id, material_id, effect)
        self.mesh.materials.append(material)
        self.materials[key] = material

    def _add_face_geometry_node(
        self, face: Face, translucent: bool = False
    ) -> collada.scene.GeometryNode:
        key = (face, translucent)
        if key in self.geometry_nodes:
            return self.geometry_nodes[key]
        # Create geometry
        id_str = f"FaceID{len(self.geometry_nodes)}"
        positions = collada.source.FloatSource(
            id_str + "_positions", face.get_vertices(), ("X", "Y", "Z")
        )
        normals = collada.source.FloatSource(
            id_str + "_normals", face.get_normal_vectors(), ("X", "Y", "Z")
        )

        geom = collada.geometry.Geometry(self.mesh, id_str, id_str, [positions, normals])
        input_list = collada.source.InputList()
        input_list.addInput(0, "VERTEX", "#" + positions.id)
        input_list.addInput(1, "NORMAL", "#" + normals.id)
        triset = geom.createTriangleSet(Face.get_triangle_indices(), input_list, _MATERIAL_SYMBOL)
        geom.primitives.append(triset)
        self.mesh.geometries.append(geom)
        # Create geometry node
        inputs = [("UVSET0", "TEXCOORD", "0")]
        self._add_material(face.color, translucent)
        material = self.materials[face.color, translucent]
        geom_node = collada.scene.GeometryNode(
            geom, [collada.scene.MaterialNode(_MATERIAL_SYMBOL, material, inputs)]
        )
        self.geometry_nodes[key] = geom_node
        return geom_node

    def _add_block_library_node(
        self,
        block_kind: BlockKind,
        pop_faces_at_directions: Iterable[SignedDirection3D] = (),
        *,
        translucent: bool = False,
    ) -> _BlockLibraryKey:
        pop_faces_at_directions = frozenset(pop_faces_at_directions) | self._pop_faces_at_directions
        key = _BlockLibraryKey(block_kind, pop_faces_at_directions, translucent)
        if key in self.block_library:
            return key
        faces = self.geometries.get_geometry(block_kind, pop_faces_at_directions)
        children = [self._add_face_geometry_node(face, translucent) for face in faces]
        key_str = str(key)
        node = collada.scene.Node(key_str, children, name=str(key.kind))
        self.mesh.nodes.append(node)
        self.block_library[key] = node
        return key

    def add_block_instance(
        self,
        transform_matrix: npt.NDArray[np.float32],
        block_kind: BlockKind,
        pop_faces_at_directions: Iterable[SignedDirection3D] = (),
        *,
        translucent: bool = False,
    ) -> None:
        """Add an instance node to the root node."""
        key = self._add_block_library_node(
            block_kind, pop_faces_at_directions, translucent=translucent
        )
        child_node = collada.scene.Node(
            f"ID{self._num_instances}",
            name=f"instance_{self._num_instances}",
            transforms=[collada.scene.MatrixTransform(transform_matrix.flatten())],
        )
        point_to_node = self.block_library[key]
        instance_node = collada.scene.NodeNode(point_to_node)
        child_node.children.append(instance_node)
        self.root_node.children.append(child_node)
        self._num_instances += 1

    def _add_surface_library_node(self, basis: Basis) -> None:
        if basis in self.surface_library:
            return
        surface = get_correlation_surface_geometry(basis)
        geometry_node = self._add_face_geometry_node(surface)
        node = collada.scene.Node(surface.color.value, [geometry_node], name=surface.color.value)
        self.mesh.nodes.append(node)
        self.surface_library[basis] = node

    def add_conditional_frame(
        self,
        matrix: npt.NDArray[np.float32],
        kind: ConditionalCubeKind,
        pop_faces_at_directions: Iterable[SignedDirection3D],
    ) -> None:
        """Mark a conditional cube with an opaque white exterior frame."""
        directions = frozenset(pop_faces_at_directions) | self._pop_faces_at_directions
        key = _BlockLibraryKey(kind, directions)
        if key not in self._frame_library:
            faces = self.geometries.get_conditional_frame_geometry(kind, directions)
            if not faces:
                return
            name = f"{key}{_CONDITIONAL_FRAME_SUFFIX}"
            node = collada.scene.Node(
                name, [self._add_face_geometry_node(face) for face in faces], name=name
            )
            self.mesh.nodes.append(node)
            self._frame_library[key] = node
        self._add_annotation_instance(self._frame_library[key], matrix, "conditional_frame")

    def add_surface_instance(self, basis: Basis, transformation: _Transformation) -> None:
        """Add a patch once, including full rectangles containing partial patches."""
        matrix = transformation.to_4d_affine_matrix()
        # Normalize signed zero so geometrically identical transforms share a key.
        matrix[matrix == 0] = 0
        key = (basis, matrix.tobytes())
        if key in self._surface_instances:
            return

        # Containment is safe for axis-aligned rectangles in the same plane.
        # Keep other shapes and partial overlaps rather than comparing their bounding boxes.
        tolerance = 1e-6
        vectors = matrix[:3, :2].astype(np.float64)
        active_axes = np.abs(vectors) > tolerance
        bounds = None
        contained_keys = []
        if np.all(active_axes.sum(axis=0) == 1) and np.all(active_axes.sum(axis=1) <= 1):
            origin = matrix[:3, 3].astype(np.float64)
            corners = np.array(
                [origin, origin + vectors[:, 0], origin + vectors[:, 1], origin + vectors.sum(1)]
            )
            bounds = np.array([corners.min(axis=0), corners.max(axis=0)])
            normal_axis = int(np.argmin(bounds[1] - bounds[0]))
            for previous_key, (previous_bounds, _) in self._surface_instances.items():
                if previous_key[0] != basis or previous_bounds is None:
                    continue
                if not np.allclose(
                    previous_bounds[:, normal_axis],
                    bounds[0, normal_axis],
                    atol=tolerance,
                    rtol=0,
                ):
                    continue
                if np.all(bounds[0] >= previous_bounds[0] - tolerance) and np.all(
                    bounds[1] <= previous_bounds[1] + tolerance
                ):
                    return
                if np.all(previous_bounds[0] >= bounds[0] - tolerance) and np.all(
                    previous_bounds[1] <= bounds[1] + tolerance
                ):
                    contained_keys.append(previous_key)
        for previous_key in contained_keys:
            _, previous_node = self._surface_instances.pop(previous_key)
            self.root_node.children.remove(previous_node)

        self._add_surface_library_node(basis)
        node = self._add_annotation_instance(
            self.surface_library[basis], matrix, "correlation_surface"
        )
        self._surface_instances[key] = (bounds, node)

    def _add_annotation_instance(
        self,
        library_node: collada.scene.Node,
        matrix: npt.NDArray[np.float32],
        suffix: str,
    ) -> collada.scene.Node:
        child_node = collada.scene.Node(
            f"ID{self._num_instances}",
            children=[collada.scene.NodeNode(library_node)],
            name=f"instance_{self._num_instances}_{suffix}",
            transforms=[collada.scene.MatrixTransform(matrix.flatten())],
        )
        self.root_node.children.append(child_node)
        self._num_instances += 1
        return child_node

    def _get_condition_link_node(self) -> collada.scene.Node:
        """Return a unit cuboid, instanced as each dash of a dependency link."""
        if self._condition_link_node is None:
            faces = []
            for direction in Direction3D.all_directions():
                face = Face(TQECColor.CONDITION_LINK, 1.0, 1.0, SignedDirection3D(direction, False))
                shift = [0.0, 0.0, 0.0]
                shift[direction.value] = 1.0
                faces.extend((face, face.shift_by(*shift).with_negated_normal_direction()))
            name = "DASH" + _CONDITION_LINK_SUFFIX
            self._condition_link_node = collada.scene.Node(
                name, [self._add_face_geometry_node(face) for face in faces], name=name
            )
            self.mesh.nodes.append(self._condition_link_node)
        return self._condition_link_node

    def add_condition_link(
        self,
        patches: list[npt.NDArray[np.float32]],
        target_center: npt.NDArray[np.float32],
        free_direction: SignedDirection3D,
        bounds: npt.NDArray[np.float64],
    ) -> None:
        """Draw a rounded exterior dashed route into the conditional cube's free cap."""
        from tqec.interop.collada._condition import (  # noqa: PLC0415
            get_condition_link_path,
            sample_dashes,
        )

        path = get_condition_link_path(patches, target_center, free_direction, bounds)
        width = 0.025
        node = self._get_condition_link_node()
        for start, end in sample_dashes(path):
            vector = end - start
            length = float(np.linalg.norm(vector))
            axis = vector / length
            reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
            side = np.cross(axis, reference)
            side /= np.linalg.norm(side)
            up = np.cross(axis, side)
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = np.column_stack((side * width, up * width, axis * length))
            matrix[:3, 3] = start - (side + up) * width / 2
            self._add_annotation_instance(node, matrix, "condition_link")


@dataclass(frozen=True)
class _Transformation:
    """Stores the translation, scale, rotation, and the composed affine matrix.

    For the reference of the transformation matrix, see https://en.wikipedia.org/wiki/Transformation_matrix.

    Attributes:
        translation: The length-3 translation vector.
        scale: The length-3 scaling vector, which is the scaling factor along each axis.
        rotation: The 3x3 rotation matrix.

    """

    translation: npt.NDArray[np.float32]
    scale: npt.NDArray[np.float32]
    rotation: npt.NDArray[np.float32]

    @staticmethod
    def from_4d_affine_matrix(mat: npt.NDArray[np.float32]) -> _Transformation:
        translation = mat[:3, 3]
        scale = np.linalg.norm(mat[:3, :3], axis=1)
        rotation = mat[:3, :3] / scale[None, :]
        return _Transformation(translation, scale, rotation)

    def to_4d_affine_matrix(self) -> npt.NDArray[np.float32]:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = self.rotation * self.scale[None, :]
        mat[:3, 3] = self.translation
        return mat
