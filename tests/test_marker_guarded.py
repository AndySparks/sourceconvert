"""marker_guarded.py: a 0-pixel image is dropped with its references, never saved
(*Obviously Awesome*, 2026-10-14/15: `cannot write empty image as JPEG`)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import marker_guarded as mg  # noqa: E402
import convert  # noqa: E402


class Img:
    def __init__(self, w, h):
        self.size = (w, h)


def test_drops_a_zero_pixel_image_and_every_reference():
    text = 'a ![](_page_3_Picture_1.jpeg) b\n![fig](_page_3_Picture_1.jpeg "t")\n![](_page_4_Picture_2.jpeg)'
    images = {"_page_3_Picture_1.jpeg": Img(0, 12), "_page_4_Picture_2.jpeg": Img(40, 30)}
    out, kept, dropped = mg.drop_empty_images(text, images)
    assert dropped == ["_page_3_Picture_1.jpeg"]
    assert list(kept) == ["_page_4_Picture_2.jpeg"]
    assert "_page_3_Picture_1" not in out and "![](_page_4_Picture_2.jpeg)" in out
    assert len(images) == 2                     # inputs untouched


def test_leaves_real_images_and_unmeasurable_ones_alone():
    text = "![](a.jpeg) ![](b.jpeg)"
    images = {"a.jpeg": Img(1, 1), "b.jpeg": object()}
    assert mg.drop_empty_images(text, images) == (text, images, [])


def test_a_name_that_is_a_prefix_of_another_is_not_confused():
    text = "![](p_1.jpeg) ![](p_1.jpeg.jpeg)"
    out, _kept, dropped = mg.drop_empty_images(text, {"p_1.jpeg": Img(0, 0), "p_1.jpeg.jpeg": Img(5, 5)})
    assert dropped == ["p_1.jpeg"] and out == " ![](p_1.jpeg.jpeg)"


def _stub(tmp_path, monkeypatch, shebang):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ms = bindir / "marker_single"
    ms.write_text(shebang + "\nprint('x')\n")
    ms.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))


def test_marker_command_runs_the_guard_under_marker_singles_interpreter(tmp_path, monkeypatch):
    _stub(tmp_path, monkeypatch, f"#!{sys.executable}")
    assert convert.marker_command() == [sys.executable, str(ROOT / "marker_guarded.py")]


def test_marker_command_falls_back_loudly_when_the_shebang_is_stale(tmp_path, monkeypatch, capsys):
    _stub(tmp_path, monkeypatch, "#!/nowhere/python3.12")
    assert convert.marker_command() == ["marker_single"]
    assert "marker guard unavailable" in capsys.readouterr().out


@pytest.mark.skipif(not (ROOT / ".venv-marker/bin/python").exists(), reason="marker venv not installed")
def test_the_patch_point_exists_in_the_installed_marker():
    """The guard patches marker.output.text_from_rendered; save_output must call it."""
    import inspect
    import subprocess
    code = ("import inspect, marker.output as o; "
            "assert hasattr(o, 'text_from_rendered'); "
            "assert 'text_from_rendered(rendered)' in inspect.getsource(o.save_output)")
    subprocess.run([str(ROOT / ".venv-marker/bin/python"), "-c", code], check=True, timeout=120)
