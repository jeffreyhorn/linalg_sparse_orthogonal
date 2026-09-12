#!/usr/bin/env python3
"""Regression tests for local-only generated API routing validation."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTING_SCRIPT = REPO_ROOT / "scripts" / "check_api_docs_routing.py"

spec = importlib.util.spec_from_file_location("check_api_docs_routing", ROUTING_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {ROUTING_SCRIPT}")
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


def copy_fixture(root: Path) -> None:
    for rel_path in routing.API_ROUTING_FILES:
        source = REPO_ROOT / rel_path
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    for rel_path in (
        "Makefile",
        "include",
        "Doxyfile",
        "docs/tutorial.md",
        "docs/cookbook.md",
        "docs/solver_selection.md",
    ):
        source = REPO_ROOT / rel_path
        dest = root / rel_path
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def assert_routing_fails_with(mutator, expected: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        mutator(root)
        try:
            routing.validate_api_routes(root)
        except routing.RoutingError as exc:
            message = str(exc)
            if expected not in message:
                raise AssertionError(f"expected {expected!r} in {message!r}") from exc
            return
        raise AssertionError("expected routing failure")


def test_current_tree_passes_routing_guard() -> None:
    routing.validate_api_routes(REPO_ROOT)


def test_fixture_passes_routing_guard() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        routing.validate_api_routes(root)


def test_missing_api_reference_route_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[docs/api_reference.md](docs/api_reference.md)",
                "docs/api_reference.md",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required API route link")


def test_missing_route_target_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "docs" / "solver_selection.md").unlink()

    assert_routing_fails_with(mutate, "links to missing API route target")


def test_generated_html_publication_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated HTML](../docs/api/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_hosted_api_publication_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted API](https://example.github.io/linalg_sparse_orthogonal/api/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_missing_local_only_text_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "The generated HTML tree is local-only generated output.",
                "The generated HTML tree is available.",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required local-only API routing text")


def test_missing_maintainer_claim_boundary_text_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "maintainer_guide.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "removes `api-docs-routing` from `make api-docs-freshness` without replacing",
                "removes generated API routing checks without replacing",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required local-only API routing text")


def test_missing_makefile_routing_dependency_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(" api-docs-routing", ""),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "api-docs-validate must depend on api-docs-routing")


def test_missing_makefile_routing_target_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "api-docs-routing:\n"
            "\t@python3 scripts/check_api_docs_routing.py\n"
            "\t@python3 tests/test_api_docs_routing.py\n\n",
            "",
        )
        path.write_text(text, encoding="utf-8")

    assert_routing_fails_with(mutate, "must define api-docs-routing")


def main() -> None:
    test_current_tree_passes_routing_guard()
    test_fixture_passes_routing_guard()
    test_missing_api_reference_route_fails_clearly()
    test_missing_route_target_fails_clearly()
    test_generated_html_publication_link_fails_clearly()
    test_hosted_api_publication_link_fails_clearly()
    test_missing_local_only_text_fails_clearly()
    test_missing_maintainer_claim_boundary_text_fails_clearly()
    test_missing_makefile_routing_dependency_fails_clearly()
    test_missing_makefile_routing_target_fails_clearly()


if __name__ == "__main__":
    main()
