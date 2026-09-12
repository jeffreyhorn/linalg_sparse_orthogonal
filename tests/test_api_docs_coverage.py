#!/usr/bin/env python3
"""Regression tests for generated API coverage diagnostics."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COVERAGE_SCRIPT = REPO_ROOT / "scripts" / "check_api_docs_coverage.py"


spec = importlib.util.spec_from_file_location("check_api_docs_coverage", COVERAGE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {COVERAGE_SCRIPT}")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_fixture(root: Path) -> tuple[Path, Path]:
    include_dir = root / "include"
    html_dir = root / "docs" / "api" / "html"

    write_file(include_dir / "sparse_matrix.h", "int sparse_matrix_symbol(void);\n")
    write_file(include_dir / "sparse_vector.h", "int sparse_vector_symbol(void);\n")
    write_file(include_dir / "sparse_version.h.in", "#define SPARSE_VERSION \"@PROJECT_VERSION@\"\n")
    write_file(html_dir / "index.html", "<!doctype html>\n")

    for stem in ("sparse__matrix_8h", "sparse__vector_8h"):
        write_file(html_dir / f"{stem}.html", "<!doctype html>\n")
        write_file(html_dir / f"{stem}_source.html", "<!doctype html>\n")

    return include_dir, html_dir


def assert_coverage_fails_with(mutator, expected: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        include_dir, html_dir = write_fixture(root)
        mutator(root, include_dir, html_dir)
        try:
            coverage.check_coverage(root, include_dir, html_dir)
        except coverage.CoverageError as exc:
            message = str(exc)
            if expected not in message:
                raise AssertionError(f"expected {expected!r} in {message!r}") from exc
            return
        raise AssertionError("expected coverage failure")


def test_complete_fixture_passes_with_checked_in_headers_only() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        include_dir, html_dir = write_fixture(root)
        counts = coverage.check_coverage(root, include_dir, html_dir)
        if counts != (2, 2, 2):
            raise AssertionError(f"unexpected coverage counts: {counts!r}")


def test_missing_html_directory_fails_clearly() -> None:
    def mutate(root: Path, include_dir: Path, html_dir: Path) -> None:
        for path in sorted(html_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        html_dir.rmdir()

    assert_coverage_fails_with(mutate, "generated API HTML directory not found")


def test_missing_index_fails_clearly() -> None:
    def mutate(root: Path, include_dir: Path, html_dir: Path) -> None:
        (html_dir / "index.html").unlink()

    assert_coverage_fails_with(mutate, "generated API index not found")


def test_missing_reference_page_identifies_header() -> None:
    def mutate(root: Path, include_dir: Path, html_dir: Path) -> None:
        (html_dir / "sparse__matrix_8h.html").unlink()

    assert_coverage_fails_with(mutate, "include/sparse_matrix.h -> missing reference page")


def test_missing_source_page_identifies_header() -> None:
    def mutate(root: Path, include_dir: Path, html_dir: Path) -> None:
        (html_dir / "sparse__vector_8h_source.html").unlink()

    assert_coverage_fails_with(mutate, "include/sparse_vector.h -> missing source page")


def test_stale_reference_page_identifies_header() -> None:
    def mutate(root: Path, include_dir: Path, html_dir: Path) -> None:
        old_time = 1_700_000_000
        new_time = 1_800_000_000
        os.utime(html_dir / "sparse__matrix_8h.html", (old_time, old_time))
        os.utime(include_dir / "sparse_matrix.h", (new_time, new_time))

    assert_coverage_fails_with(mutate, "include/sparse_matrix.h -> stale reference page")


def test_stale_source_page_identifies_header() -> None:
    def mutate(root: Path, include_dir: Path, html_dir: Path) -> None:
        old_time = 1_700_000_000
        new_time = 1_800_000_000
        os.utime(html_dir / "sparse__vector_8h_source.html", (old_time, old_time))
        os.utime(include_dir / "sparse_vector.h", (new_time, new_time))

    assert_coverage_fails_with(mutate, "include/sparse_vector.h -> stale source page")


def main() -> None:
    test_complete_fixture_passes_with_checked_in_headers_only()
    test_missing_html_directory_fails_clearly()
    test_missing_index_fails_clearly()
    test_missing_reference_page_identifies_header()
    test_missing_source_page_identifies_header()
    test_stale_reference_page_identifies_header()
    test_stale_source_page_identifies_header()


if __name__ == "__main__":
    main()
