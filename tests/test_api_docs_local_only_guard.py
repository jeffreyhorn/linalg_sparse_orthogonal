#!/usr/bin/env python3
"""Regression tests for the generated API local-only guard."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_api_docs_local_only.sh"


def run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_fixture(root: Path) -> None:
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    (root / "docs" / "api" / "html").mkdir(parents=True)
    (root / ".github" / "workflows").mkdir(parents=True)

    (root / "scripts" / SCRIPT.name).write_text(
        SCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / ".gitignore").write_text("docs/api/\n", encoding="utf-8")
    (root / "docs" / "api" / "html" / "index.html").write_text(
        "<!doctype html>\n",
        encoding="utf-8",
    )
    (root / "Doxyfile").write_text(
        "INPUT                  = include/\n"
        "FILE_PATTERNS          = *.h\n"
        "RECURSIVE              = NO\n"
        "OUTPUT_DIRECTORY       = docs/api\n"
        "GENERATE_HTML          = YES\n"
        "HTML_OUTPUT            = html\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "make api-docs-freshness # selected local Doxygen freshness plus "
        "local-only staging guard\n",
        encoding="utf-8",
    )
    (root / "docs" / "api_reference.md").write_text(
        "The generated HTML tree is local-only generated output.\n"
        "It is not a hosted or source-controlled publication surface.\n",
        encoding="utf-8",
    )
    (root / "docs" / "maintainer_guide.md").write_text(
        "The maintained Sprint 179 product decision keeps this tree\n"
        "local-only and ignored rather than committed, hosted, or "
        "artifact-published.\n"
        "local generated output under `docs/api/html/` is not "
        "source-controlled,\n"
        "hosted, artifact-published, or release evidence.\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "ci.yml").write_text(
        "name: ci\n"
        "on: [push]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: make test\n",
        encoding="utf-8",
    )

    result = run(["git", "init", "-q"], root)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    return run(["bash", "scripts/check_api_docs_local_only.sh"], root)


def assert_guard_fails_with(mutator, expected: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_fixture(root)
        mutator(root)
        result = run_guard(root)
        if result.returncode == 0:
            raise AssertionError("expected guard failure")
        message = result.stdout + result.stderr
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}")


def test_current_tree_passes_guard() -> None:
    result = run_guard(REPO_ROOT)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def test_fixture_passes_guard() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_fixture(root)
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_generated_api_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: ls docs/api/html\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "generated API HTML output path")


def test_workflow_publication_semantics_fail_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: docs/api/html\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_missing_local_only_wording_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "docs" / "api_reference.md").write_text(
            "Generated API docs are available.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "local-only generated output")


def main() -> None:
    test_current_tree_passes_guard()
    test_fixture_passes_guard()
    test_workflow_generated_api_path_fails_clearly()
    test_workflow_publication_semantics_fail_clearly()
    test_missing_local_only_wording_fails_clearly()


if __name__ == "__main__":
    main()
