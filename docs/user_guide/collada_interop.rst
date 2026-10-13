COLLADA Interoperability
========================

This page demonstrates how to interoperate :code:`tqec` with `COLLADA <https://en.wikipedia.org/wiki/COLLADA>`_ ( :code:`.dae`) files. COLLADA files can be
imported/exported to/from `SketchUp <https://en.wikipedia.org/wiki/SketchUp>`_, which is a commonly used 3D modeling tool in QEC papers.

:code:`tqec` relies on `PyCollada <https://github.com/pycollada/pycollada>`_ under the hood to realize the interoperability with COLLADA files.

Import COLLADA model
---------------------

A common workflow starting from SketchUp is:

1. Open the `template file <https://github.com/QCHackers/tqec/blob/main/assets/template.skp>`_ in SketchUp.
2. Construct the model representing the logical computation.
3. Export the model as a COLLADA(:code:`.dae`) file from SketchUp.
4. Import the COLLADA file to :code:`tqec` as a :code:`BlockGraph`, which can be compiled to circuits.

:code:`tqec` provides the function :code:`tqec.interop.collada.read_block_graph_from_dae_file` to import a COLLADA file as a :code:`BlockGraph`. Or you can
call :code:`BlockGraph.from_dae_file` directly.

.. jupyter-execute::

    from tqec import BlockGraph

    graph = BlockGraph.from_dae_file("../assets/logical_cnot.dae")

Export COLLADA model
---------------------

If you start with building the logical computation in :code:`tqec` and build a :code:`BlockGraph`, you can export the :code:`BlockGraph` to a COLLADA file by
calling :code:`BlockGraph.to_dae_file`.

.. code-block:: python

    graph.to_dae_file("logical_cnot.dae")

Display COLLADA model
---------------------

:code:`tqec` provides the function :code:`tqec.interop.collada.display_collada_model` to view the COLLADA model as html and render it with `three.js <https://threejs.org/>`_.
It can be used in IPython environment to display and play with the model interactively. You can also call :code:`BlockGraph.view_as_html` directly to first
convert it to a COLLADA model then display it.

.. jupyter-execute::

    graph.view_as_html()

Additionally, you can attach the correlation surface to the model to visualize what the logical observable looks like:

.. jupyter-execute::

    correlation_surfaces = graph.find_correlation_surfaces()
    graph.view_as_html(
        pop_faces_at_directions=("-Y",),
        show_correlation_surface=correlation_surfaces[0],
    )

Visualize conditional cubes
---------------------------

Conditional cubes appear as two colored halves, one for each branch. Each conditional
cube must be a leaf with exactly one pipe compatible with both branches. HTML viewing
and DAE export reject isolated conditional cubes, conditional cubes with multiple
pipes, and incompatible connections, even when condition display is disabled.

An opaque white (``#FFFFFF``) outer frame distinguishes conditional cubes from
ordinary cubes while preserving the face basis colors. The frame follows the outer
cube edges, without an added split seam. The frame is also shown with
``show_conditions=False``.

The plane dividing the halves contains the pipe direction. The split coordinate is
Y for an X pipe, Z for a Y pipe, and X for a Z pipe. Branch 0 occupies the lower half
of that coordinate and branch 1 the upper half. Matching face colors are merged.
The face opposite the connecting pipe displays both branch colors regardless of
the pipe's sign. Explicitly removing that face with ``pop_faces_at_directions`` can
hide the distinction. A Y branch is shown in green on a full-size symbolic cube;
it does not use the height or offset of an ordinary Y half cube.

By default, ``view_as_html()`` and ``to_dae_file()`` also show each conditional cube's
controlling partial correlation surface. Only cubes and pipes containing those
surfaces become translucent; the rest of the structure stays opaque. Gray dashed
connectors follow rounded routes outside the structure and enter the conditional
cubes at their free caps, opposite their connecting pipes. Each route starts near
the area-weighted center of its condition surface, projected onto an actual patch
for bent or disconnected surfaces. The exterior segment makes the dependency easier
to trace, although other geometry can still hide it from some camera angles.
Partial surfaces can have open ends and terminate at earlier conditional cubes without selecting
their branches. Shared surface patches are displayed once, while each controlled
cube keeps its own connector. Singleton conditions appear as small basis-colored
tiles; empty conditions add no annotation. Referenced cubes and pipes must exist
in the graph, otherwise condition display raises ``TQECError``.

For example, the following graph measures the upper-right cube in either the
``ZXZ`` or ``ZXX`` branch, controlled by an X-basis partial surface on the earlier
spatial pipe:

.. jupyter-execute::

    from tqec import Basis, BlockGraph
    from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
    from tqec.utils.position import Position3D

    a0, a1 = Position3D(0, 0, 0), Position3D(0, 0, 1)
    b1, measurement = Position3D(1, 0, 1), Position3D(1, 0, 2)
    condition = CorrelationSurface(
        frozenset({ZXEdge(ZXNode(a1, Basis.X), ZXNode(b1, Basis.X))})
    )

    conditional_graph = BlockGraph("conditional measurement")
    for position in (a0, a1, b1):
        conditional_graph.add_cube(position, "ZXZ")
    conditional_graph.add_cube(measurement, "ZXZ_ZXX", condition=condition)
    conditional_graph.add_pipe(a0, a1)
    conditional_graph.add_pipe(a1, b1)
    conditional_graph.add_pipe(b1, measurement)
    conditional_graph.validate()
    conditional_graph.view_as_html()

Use the keyword-only ``show_conditions=False`` argument to display opaque branch
halves without the automatic surfaces and connectors. An explicitly supplied
``show_correlation_surface`` remains available independently of this option.

.. jupyter-execute::

    conditional_graph.view_as_html(show_conditions=False)

DAE export uses the same visualization options:

.. code-block:: python

    conditional_graph.to_dae_file("conditional_measurement.dae")
    conditional_graph.to_dae_file("branches_only.dae", show_conditions=False)

Conditional DAE files preserve the visualization but cannot be imported back as
block graphs because the condition data cannot be reconstructed from the geometry.
Use JSON to save and restore conditional computations:

.. code-block:: python

    conditional_graph.to_json("conditional_measurement.json")
    restored_graph = BlockGraph.from_json(
        "conditional_measurement.json", graph_name=conditional_graph.name
    )
