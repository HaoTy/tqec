import html
import json
from pathlib import Path
from typing import Any

import pytest

from tqec.visualisation.patches.html import PatchHTMLViewer, display_patch_scene


def _scene(name: str = "Memory") -> dict[str, Any]:
    return {
        "name": name,
        "k": 1,
        "d": 3,
        "cycles": [],
        "coarse": [],
        "outlines": [],
        "observables": [],
        "notices": [],
    }


def test_html_round_trip_and_export(tmp_path: Path) -> None:
    scene = _scene("Memory · Z")
    path = tmp_path / "patches.html"
    viewer = display_patch_scene(scene, path)
    assert isinstance(viewer, PatchHTMLViewer)
    assert str(viewer) == viewer.html_str == path.read_text(encoding="utf-8")
    payload = str(viewer).split('<script id="patch-scene" type="application/json">')[1]
    payload = payload.split("</script>", 1)[0]
    assert json.loads(payload) == scene
    assert "three@0.138.0" in str(viewer)
    assert "async function startPatchViewer()" in str(viewer)


def test_untrusted_labels_do_not_escape_json_script() -> None:
    attack = '</script><script>alert("x")</script>&\u2028'
    viewer = PatchHTMLViewer(_scene(attack))
    assert attack not in viewer.html_str
    payload = viewer.html_str.split('<script id="patch-scene" type="application/json">')[1]
    payload = payload.split("</script>", 1)[0]
    assert "<" not in payload
    assert "&" not in payload
    assert json.loads(payload)["name"] == attack


def test_notebook_documents_are_isolated_and_escaped() -> None:
    first = PatchHTMLViewer(_scene('First "quoted" graph'))
    second = PatchHTMLViewer(_scene("Second graph"))
    frame = first._repr_html_()
    assert frame.startswith("<iframe ")
    assert "<script" not in frame
    assert frame.count("srcdoc=") == 1
    srcdoc = frame.split('srcdoc="', 1)[1].rsplit('"', 1)[0]
    assert html.unescape(srcdoc) == first.html_str
    assert first.html_str != second.html_str
    assert "Second graph" not in first.html_str


def test_reject_nonfinite_scene_coordinates() -> None:
    scene = _scene()
    scene["coarse"] = [{"center": [0, 0, float("nan")]}]
    with pytest.raises(ValueError, match="Out of range float"):
        PatchHTMLViewer(scene)


def test_viewer_replacement_tokens_in_labels_are_preserved() -> None:
    name = "__VIEWER_JS__ __SCENE_JSON__"
    viewer = PatchHTMLViewer(_scene(name))
    payload = viewer.html_str.split('<script id="patch-scene" type="application/json">')[1]
    payload = payload.split("</script>", 1)[0]
    assert json.loads(payload)["name"] == name
