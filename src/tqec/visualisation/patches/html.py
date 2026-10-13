"""Interactive, notebook-compatible HTML for hybrid physical and block scenes."""

from __future__ import annotations

import html
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TQEC surface-code patches</title>
  <script type="importmap">
    {"imports": {
      "three": "https://unpkg.com/three@0.138.0/build/three.module.js",
      "three-orbitcontrols":
        "https://unpkg.com/three@0.138.0/examples/jsm/controls/OrbitControls.js"
    }}
  </script>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; }
    body { color: #243445; font: 13px/1.45 system-ui, sans-serif; background: #f4f7fa; }
    main { height: 100%; display: flex; flex-direction: column; }
    header { padding: 12px 16px 8px; border-bottom: 1px solid #d8e0e8; background: white; }
    h1 { margin: 0; font-size: 17px; font-weight: 650; overflow-wrap: anywhere; }
    #summary { color: #536574; font-size: 12px; }
    #workspace { display: flex; flex: 1; min-height: 0; }
    aside { width: 230px; flex-shrink: 0; padding: 12px; overflow-y: auto;
            border-right: 1px solid #d8e0e8; background: #fff; }
    fieldset { padding: 0; border: 0; margin: 0 0 14px; min-width: 0; }
    legend { font-weight: 650; margin-bottom: 6px; }
    label { display: block; margin: 4px 0; }
    label.inline { display: flex; gap: 6px; align-items: center; }
    input[type=range] { width: 100%; margin: 4px 0; accent-color: #32619a; }
    input[type=checkbox] { accent-color: #32619a; }
    select, button { font: inherit; color: inherit; border: 1px solid #bfccd7;
                     border-radius: 5px; background: white; padding: 5px 7px; }
    select { width: 100%; }
    button { cursor: pointer; }
    button:hover { background: #edf2f7; }
    #round-label, .help { font-size: 11px; color: #596b7b; }
    #round-label { min-height: 2.7em; }
    #viewport { position: relative; flex: 1; min-width: 0; min-height: 150px; }
    canvas { display: block; width: 100%; height: 100%; touch-action: none; }
    #status { position: absolute; top: 16px; left: 16px; right: 16px;
              border-radius: 6px; padding: 12px; background: #fff; z-index: 1; }
    #tooltip { position: absolute; display: none; pointer-events: none; z-index: 2;
               white-space: pre-wrap; overflow-wrap: anywhere; max-width: min(390px, 85%);
               max-height: 60%; overflow: hidden; border: 1px solid #b9c7d4;
               border-radius: 6px; background: #fffffff2; padding: 8px 10px;
               box-shadow: 0 3px 15px #26374726; font-size: 12px; }
    #legend { position: absolute; bottom: 10px; left: 12px; right: 12px;
              display: flex; flex-wrap: wrap; gap: 5px 13px; pointer-events: none;
              font-size: 11px; color: #3b4a59; }
    #legend span { background: #ffffffd9; padding: 2px 5px; border-radius: 3px; }
    .swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2px;
              margin-right: 4px; vertical-align: middle; }
    #notices { margin: 0; padding: 8px 12px 8px 28px; border-top: 1px solid #d8e0e8;
               background: #fff9e9; max-height: 100px; overflow: auto; font-size: 12px; }
    #notices:empty { display: none; }
    @media (max-width: 600px) {
      aside { width: 180px; padding: 9px; }
      header { padding: 8px 12px; }
      #legend { gap: 3px; }
    }
  </style>
</head>
<body>
<main>
  <header><h1 id="title">Surface-code patches</h1><div id="summary"></div></header>
  <div id="workspace">
    <aside aria-label="Visualization controls">
      <fieldset>
        <legend>Rounds</legend>
        <label for="first-round">First displayed round</label>
        <input id="first-round" type="range" min="0" max="0" value="0">
        <label for="last-round">Last displayed round</label>
        <input id="last-round" type="range" min="0" max="0" value="0">
        <label class="inline"><input id="single-round" type="checkbox">Single round</label>
        <div id="round-label" aria-live="polite"></div>
        <button id="all-rounds" type="button">Show all</button>
      </fieldset>
      <fieldset>
        <legend>Display</legend>
        <label for="spacing">Layer spacing</label>
        <input id="spacing" type="range" min="0.25" max="3" step="0.05" value="2">
        <label for="opacity">Patch opacity</label>
        <input id="opacity" type="range" min="0.05" max="0.85" step="0.01" value="0.24">
        <label class="inline"><input id="show-data" type="checkbox" checked>Data qubits</label>
        <label class="inline"><input id="show-syndrome" type="checkbox" checked>
          Syndrome ancillas</label>
        <label class="inline"><input id="show-checks" type="checkbox" checked>
          Measured checks</label>
        <label class="inline"><input id="show-changed" type="checkbox">Changed stabilizers</label>
        <label class="inline"><input id="show-coarse" type="checkbox" checked>Block fallback</label>
        <label class="inline"><input id="show-outlines" type="checkbox" checked>
          Block outlines</label>
        <button id="fit" type="button">Fit view</button>
      </fieldset>
      <fieldset>
        <legend>Correlation surface</legend>
        <label for="observable" class="help">Select an observable</label>
        <select id="observable" aria-label="Observable"></select>
        <label class="inline"><input id="show-logical" type="checkbox" checked>
          Logical operators</label>
        <label class="inline"><input id="show-deformation" type="checkbox">
          Logical deformation</label>
        <div id="observable-detail" class="help"></div>
      </fieldset>
      <div class="help">Drag to orbit · scroll to zoom · right-drag to pan.
        Hover over a qubit or operator for its support and round.
        The vertical axis is time; unknown operations retain their block-time placement.</div>
    </aside>
    <div id="viewport" aria-label="3D surface-code scene">
      <div id="status" role="status">Loading 3D viewer… Internet access and JavaScript
        are required to load Three.js.</div>
      <div id="tooltip" role="tooltip"></div>
      <div id="legend">
        <span><i class="swatch" style="background:#ff7f7f"></i>X</span>
        <span><i class="swatch" style="background:#63c676"></i>Y</span>
        <span><i class="swatch" style="background:#7396ff"></i>Z</span>
        <span>● data · ◆ syndrome</span>
        <span>□ preparation · ○ readout</span>
        <span><i class="swatch" style="background:#dea521"></i>deformation factor</span>
        <span style="color:#ad6700">◆ tracking boundary</span>
        <span><i class="swatch" style="border:1px dashed #6d7988"></i>coarse region</span>
        <span style="color:#12814b">+ added</span>
        <span style="color:#bb3e31">&minus; removed</span>
      </div>
    </div>
  </div>
  <ul id="notices" aria-label="Visualization notices"></ul>
</main>
<script id="patch-scene" type="application/json">__SCENE_JSON__</script>
<script type="module">__VIEWER_JS__</script>
</body>
</html>
"""


class PatchHTMLViewer:
    """A self-contained scene document, embeddable in an isolated notebook iframe.

    The scene data and viewer code are embedded in the document. Three.js and its
    orbit controls are loaded from the same pinned CDN version as the block viewer.

    """

    def __init__(self, scene: dict[str, Any]) -> None:
        """Create the HTML document for a JSON-serializable hybrid scene."""
        # Escaping '<' prevents arbitrary scene labels from terminating the JSON
        # script element. Do not HTML-escape JSON: script elements contain raw text.
        scene_json = json.dumps(scene, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        scene_json = scene_json.replace("&", "\\u0026").replace("<", "\\u003c")
        scene_json = scene_json.replace(">", "\\u003e")
        viewer_js = files("tqec.visualisation.patches").joinpath("viewer.js").read_text("utf-8")
        self.html_str = _HTML.replace("__VIEWER_JS__", viewer_js).replace(
            "__SCENE_JSON__", scene_json
        )

    def __str__(self) -> str:
        return self.html_str

    def _repr_html_(self) -> str:
        return (
            '<iframe title="TQEC surface-code patches" '
            'style="width:100%;height:680px;resize:both;border:1px solid #d8e0e8;" '
            f'srcdoc="{html.escape(self.html_str, quote=True)}"></iframe>'
        )


def display_patch_scene(
    scene: dict[str, Any], write_html_filepath: str | Path | None = None
) -> PatchHTMLViewer:
    """Create an interactive hybrid patch viewer and optionally write its HTML.

    Args:
        scene: JSON-serializable physical patches and block fallback geometry.
        write_html_filepath: Optional standalone HTML output path.

    Returns:
        A viewer that displays directly in IPython-compatible notebooks.

    """
    viewer = PatchHTMLViewer(scene)
    if write_html_filepath is not None:
        Path(write_html_filepath).write_text(viewer.html_str, encoding="utf-8")
    return viewer
