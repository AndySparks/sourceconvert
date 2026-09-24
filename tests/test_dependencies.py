"""Tests for check_dependencies gating."""
import sys

import pytest

import convert


def test_check_dependencies_pymupdf4llm_on_old_python():
    """On Python 3.9, check_dependencies should raise with a helpful message."""
    if sys.version_info >= (3, 10):
        pytest.skip("Test only meaningful on Python 3.9")
    with pytest.raises(convert.DependencyError, match=r"pymupdf4llm"):
        convert.check_dependencies("pymupdf4llm")


def test_check_dependencies_pymupdf4llm_on_new_python():
    """On Python 3.10+, check_dependencies should succeed if pymupdf4llm is installed."""
    if sys.version_info < (3, 10):
        pytest.skip("Test only meaningful on Python 3.10+")
    try:
        import pymupdf4llm  # noqa: F401
    except ImportError:
        pytest.skip("pymupdf4llm not installed in this interpreter")
    convert.check_dependencies("pymupdf4llm")


def test_check_dependencies_docling_on_old_python():
    if sys.version_info >= (3, 10):
        pytest.skip("Test only meaningful on Python 3.9")
    with pytest.raises(convert.DependencyError, match=r"docling"):
        convert.check_dependencies("docling")


def test_cli_accepts_pymupdf4llm_method():
    """main() should accept --method pymupdf4llm without argparse error."""
    import sys as _sys

    old_argv = _sys.argv
    try:
        _sys.argv = ["convert.py", "nonexistent.pdf", "--method", "pymupdf4llm", "--skip-check"]
        with pytest.raises(SystemExit):
            convert.main()
    finally:
        _sys.argv = old_argv


def test_cli_accepts_docling_method():
    import sys as _sys

    old_argv = _sys.argv
    try:
        _sys.argv = ["convert.py", "nonexistent.pdf", "--method", "docling", "--skip-check"]
        with pytest.raises(SystemExit):
            convert.main()
    finally:
        _sys.argv = old_argv


def test_pick_ocr_backend_prefers_marker_when_available(monkeypatch):
    """When marker is importable, pick_ocr_backend returns 'marker'."""
    monkeypatch.setattr(convert, "_marker_available", lambda: True)
    assert convert.pick_ocr_backend() == "marker"


def test_pick_ocr_backend_falls_back_to_ocr(monkeypatch):
    """When marker is not available, pick_ocr_backend returns 'ocr'."""
    monkeypatch.setattr(convert, "_marker_available", lambda: False)
    assert convert.pick_ocr_backend() == "ocr"


def test_check_dependencies_marker_accepts_binary_without_import(monkeypatch, tmp_path):
    """marker runs as the `marker_single` subprocess, never in-process.

    ingest-batch.sh runs convert.py under `.venv` (no marker-pdf) with
    `.venv-marker/bin` on PATH. The --auto-ocr retry calls
    check_dependencies("marker") unconditionally; requiring `import marker`
    there failed every scanned-text-layer retry on 2026-09-24 even though
    pick_ocr_backend() had just chosen marker from the same binary.
    """
    import builtins
    import shutil

    fake_bin = tmp_path / "marker_single"
    fake_bin.write_text(f"#!{sys.executable}\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(shutil, "which",
                        lambda name: str(fake_bin) if name == "marker_single" else None)

    real_import = builtins.__import__

    def no_marker(name, *args, **kwargs):
        if name == "marker" or name.startswith("marker."):
            raise ImportError("No module named 'marker'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_marker)
    monkeypatch.setitem(sys.modules, "marker", None)

    assert convert._marker_available() is True
    convert.check_dependencies("marker")  # must not raise


def test_check_dependencies_marker_still_fails_without_binary(monkeypatch, tmp_path):
    """The other half: no binary anywhere is still a DependencyError."""
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    with pytest.raises(convert.DependencyError):
        convert.check_dependencies("marker")
