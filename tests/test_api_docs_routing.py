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


def test_required_route_in_fenced_code_does_not_satisfy_contract() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[docs/api_reference.md](docs/api_reference.md)",
                "`docs/api_reference.md`",
            )
            + "\n```md\n[docs/api_reference.md](docs/api_reference.md)\n```\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required API route link")


def test_required_route_in_indented_fenced_code_does_not_satisfy_contract() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[docs/api_reference.md](docs/api_reference.md)",
                "`docs/api_reference.md`",
            )
            + "\n   ```md\n[docs/api_reference.md](docs/api_reference.md)\n   ```\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required API route link")


def test_required_route_in_indented_code_does_not_satisfy_contract() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[docs/api_reference.md](docs/api_reference.md)",
                "`docs/api_reference.md`",
            )
            + "\n    [docs/api_reference.md](docs/api_reference.md)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required API route link")


def test_required_route_in_html_comment_does_not_satisfy_contract() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[docs/api_reference.md](docs/api_reference.md)",
                "`docs/api_reference.md`",
            )
            + "\n<!-- [docs/api_reference.md](docs/api_reference.md) -->\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required API route link")


def test_required_route_in_inline_code_does_not_satisfy_contract() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[docs/api_reference.md](docs/api_reference.md)",
                "`[docs/api_reference.md](docs/api_reference.md)`",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required API route link")


def test_escaped_required_route_label_does_not_satisfy_contract() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[docs/api_reference.md](docs/api_reference.md)",
                "\\[docs/api_reference.md](docs/api_reference.md)",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required API route link")


def test_angle_wrapped_required_route_with_title_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "[docs/api_reference.md](docs/api_reference.md)",
                '[API reference](<docs/api_reference.md> "source-controlled API reference")',
            ),
            encoding="utf-8",
        )
        routing.validate_api_routes(root)


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


def test_nested_label_generated_html_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[generated [API]](docs/api/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_balanced_parentheses_generated_html_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated](x(y)/../api/html/index.html)\n",
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


def test_github_pages_publication_link_without_api_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https://example.github.io/linalg_sparse_orthogonal/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_bare_hosted_api_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\nHosted docs: https://example.github.io/linalg_sparse_orthogonal/\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_autolinked_hosted_api_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n<https://example.github.io/linalg_sparse_orthogonal/>\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_generic_uri_autolink_generated_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n<file:docs/api/html/index.html>\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_generic_uri_autolink_hosted_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n<ftp://docs.example/api/>\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_html_href_hosted_api_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + '\n<a href="https://example.github.io/linalg_sparse_orthogonal/">Hosted docs</a>\n',
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_html_href_with_spacing_generated_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + '\n<a class="api-link" href = "api/html/index.html">Generated HTML</a>\n',
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_unquoted_html_href_generated_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + '\n<a class="api-link" href=api/html/index.html>Generated HTML</a>\n',
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_html_entity_href_generated_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + '\n<a href="docs&#x2F;api&#x2F;html/index.html">Generated HTML</a>\n',
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_markdown_entity_generated_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated](docs&#x2F;api&#x2F;html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_empty_html_href_is_ignored_without_traceback() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8") + '\n<a href="">Empty local placeholder</a>\n',
            encoding="utf-8",
        )
        routing.validate_api_routes(root)


def test_protocol_relative_hosted_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](//example.github.io/linalg_sparse_orthogonal/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_uppercase_hosted_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n[Hosted](HTTPS://example.github.io/api/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_arbitrary_external_hosted_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https://docs.example.com/linalg_sparse_orthogonal/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_percent_encoded_external_api_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https://reference.example.com/docs%2Fapi%2Fhtml/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_readthedocs_hosted_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https://linalg-sparse-orthogonal.readthedocs.io/en/latest/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_gitlab_hosted_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https://linalg-sparse-orthogonal.gitlab.io/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_netlify_hosted_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https://linalg-sparse-orthogonal.netlify.app/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_custom_domain_hosted_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https://reference.linalg-sparse-orthogonal.example/docs/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_ftp_hosted_publication_url_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](ftp://docs.example.com/api/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_file_scheme_generated_html_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated HTML](file:docs/api/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_repository_external_link_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Repository](https://github.com/jeffreyhorn/linalg_sparse_orthogonal)\n",
            encoding="utf-8",
        )
        routing.validate_api_routes(root)


def test_backslash_escaped_scheme_hosted_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https\\://example.com/docs/api/html/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_unrelated_external_link_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\nProject contact: <mailto:maintainers@example.com>\n"
            + "\n[Dependency](https://www.netlib.org/lapack/)\n",
            encoding="utf-8",
        )
        routing.validate_api_routes(root)


def test_unrelated_https_link_is_not_reclassified_as_protocol_relative() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n[Dependency](https://www.netlib.org/lapack/)\n",
            encoding="utf-8",
        )
        targets = routing.publication_link_targets(path.read_text(encoding="utf-8"))
        if "//www.netlib.org/lapack/" in targets:
            raise AssertionError("https URL was also emitted as protocol-relative target")
        routing.validate_api_routes(root)


def test_external_url_with_incidental_api_substring_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Capital city example](https://www.netlib.org/capitals)\n",
            encoding="utf-8",
        )
        routing.validate_api_routes(root)


def test_external_url_with_incidental_capitals_substring_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        copy_fixture(root)
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Capital letters](https://www.netlib.org/capitals)\n",
            encoding="utf-8",
        )
        routing.validate_api_routes(root)


def test_reference_generated_api_publication_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[API HTML]: ../docs/api/html/index.html\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_angle_wrapped_inline_generated_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n[Generated HTML](<api/html/index.html>)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_angle_wrapped_inline_generated_api_link_with_title_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + '\n[Generated HTML](<api/html/index.html> "title")\n',
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_angle_wrapped_reference_generated_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[API HTML]: <../docs/api/html/index.html>\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_indented_reference_generated_api_publication_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n   [API HTML]: ../docs/api/html/index.html\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_generated_api_fragment_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated API root](../docs/api#overview)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_generated_api_query_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n[Generated API root](../docs/api?view=full)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_relative_generated_html_link_fails_after_resolution() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated HTML](api/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_linked_tutorial_generated_html_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "tutorial.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated HTML](api/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_linked_cookbook_hosted_api_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "cookbook.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Hosted docs](https://linalg-sparse-orthogonal.netlify.app/)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_linked_solver_selection_generated_html_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "solver_selection.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated HTML](api/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_percent_encoded_generated_html_link_fails_after_resolution() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n[Generated HTML](docs/%61pi/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_backslash_escaped_generated_html_link_fails_after_resolution() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated](docs\\/api\\/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_root_relative_generated_html_link_fails_after_resolution() -> None:
    def mutate(root: Path) -> None:
        path = root / "README.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n[Generated HTML](/docs/api/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "unsupported generated or hosted API publication target")


def test_escaping_local_generated_html_link_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Generated HTML](../../docs/api/html/index.html)\n",
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "link escapes repository")


def test_missing_route_fragment_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "INSTALL.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "## Support Readiness Matrix",
                "## Support Matrix",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "links to missing API route fragment")


def test_route_fragment_in_fenced_heading_does_not_satisfy_contract() -> None:
    def mutate(root: Path) -> None:
        path = root / "INSTALL.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "## Support Readiness Matrix",
                "## Support Matrix\n\n```md\n## Support Readiness Matrix\n```\n",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "links to missing API route fragment")


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


def test_hidden_local_only_text_does_not_satisfy_contract() -> None:
    def mutate(root: Path) -> None:
        path = root / "docs" / "api_reference.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "The generated HTML tree is local-only generated output.",
                "The generated HTML tree is available.\n\n"
                "```md\nThe generated HTML tree is local-only generated output.\n```\n",
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
            path.read_text(encoding="utf-8").replace(
                "api-docs-validate: docs-check api-docs-local-only api-docs-routing",
                "api-docs-validate: docs-check api-docs-local-only",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "api-docs-validate must depend on api-docs-routing")


def test_missing_makefile_docs_check_dependency_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "api-docs-validate: docs-check api-docs-local-only api-docs-routing",
                "api-docs-validate: api-docs-local-only api-docs-routing",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required prerequisite(s): docs-check")


def test_missing_makefile_local_only_dependency_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "api-docs-validate: docs-check api-docs-local-only api-docs-routing",
                "api-docs-validate: docs-check api-docs-routing",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required prerequisite(s): api-docs-local-only")


def test_makefile_validate_routing_only_dependency_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "api-docs-validate: docs-check api-docs-local-only api-docs-routing",
                "api-docs-validate: api-docs-routing",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "missing required prerequisite(s): docs-check")


def test_makefile_routing_extra_dependency_name_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "api-docs-validate: docs-check api-docs-local-only api-docs-routing",
                "api-docs-validate: docs-check api-docs-local-only api-docs-routing-extra",
            ),
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


def test_missing_makefile_freshness_dependency_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "api-docs-freshness: api-docs-validate",
                "api-docs-freshness: docs",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "api-docs-freshness must depend on api-docs-validate")


def test_parallel_docs_check_generation_before_coverage_wiring_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "docs-check: docs\n\t@$(MAKE) api-docs-coverage",
                "docs-check: docs api-docs-coverage",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "docs-check must serialize docs before api-docs-coverage")


def test_makefile_freshness_extra_dependency_name_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        path = root / "Makefile"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "api-docs-freshness: api-docs-validate",
                "api-docs-freshness: api-docs-validate-extra",
            ),
            encoding="utf-8",
        )

    assert_routing_fails_with(mutate, "api-docs-freshness must depend on api-docs-validate")


def main() -> None:
    test_current_tree_passes_routing_guard()
    test_fixture_passes_routing_guard()
    test_missing_api_reference_route_fails_clearly()
    test_required_route_in_fenced_code_does_not_satisfy_contract()
    test_required_route_in_indented_fenced_code_does_not_satisfy_contract()
    test_required_route_in_indented_code_does_not_satisfy_contract()
    test_required_route_in_html_comment_does_not_satisfy_contract()
    test_required_route_in_inline_code_does_not_satisfy_contract()
    test_escaped_required_route_label_does_not_satisfy_contract()
    test_angle_wrapped_required_route_with_title_is_allowed()
    test_missing_route_target_fails_clearly()
    test_generated_html_publication_link_fails_clearly()
    test_nested_label_generated_html_link_fails_clearly()
    test_balanced_parentheses_generated_html_link_fails_clearly()
    test_hosted_api_publication_link_fails_clearly()
    test_github_pages_publication_link_without_api_path_fails_clearly()
    test_bare_hosted_api_publication_url_fails_clearly()
    test_autolinked_hosted_api_publication_url_fails_clearly()
    test_generic_uri_autolink_generated_api_link_fails_clearly()
    test_generic_uri_autolink_hosted_api_link_fails_clearly()
    test_html_href_hosted_api_publication_url_fails_clearly()
    test_html_href_with_spacing_generated_api_link_fails_clearly()
    test_unquoted_html_href_generated_api_link_fails_clearly()
    test_html_entity_href_generated_api_link_fails_clearly()
    test_markdown_entity_generated_api_link_fails_clearly()
    test_empty_html_href_is_ignored_without_traceback()
    test_protocol_relative_hosted_publication_url_fails_clearly()
    test_uppercase_hosted_publication_url_fails_clearly()
    test_arbitrary_external_hosted_publication_url_fails_clearly()
    test_percent_encoded_external_api_publication_url_fails_clearly()
    test_readthedocs_hosted_publication_url_fails_clearly()
    test_gitlab_hosted_publication_url_fails_clearly()
    test_netlify_hosted_publication_url_fails_clearly()
    test_custom_domain_hosted_publication_url_fails_clearly()
    test_ftp_hosted_publication_url_fails_clearly()
    test_file_scheme_generated_html_link_fails_clearly()
    test_repository_external_link_is_allowed()
    test_backslash_escaped_scheme_hosted_api_link_fails_clearly()
    test_unrelated_external_link_is_allowed()
    test_unrelated_https_link_is_not_reclassified_as_protocol_relative()
    test_external_url_with_incidental_api_substring_is_allowed()
    test_external_url_with_incidental_capitals_substring_is_allowed()
    test_reference_generated_api_publication_link_fails_clearly()
    test_angle_wrapped_inline_generated_api_link_fails_clearly()
    test_angle_wrapped_inline_generated_api_link_with_title_fails_clearly()
    test_angle_wrapped_reference_generated_api_link_fails_clearly()
    test_indented_reference_generated_api_publication_link_fails_clearly()
    test_generated_api_fragment_link_fails_clearly()
    test_generated_api_query_link_fails_clearly()
    test_relative_generated_html_link_fails_after_resolution()
    test_linked_tutorial_generated_html_link_fails_clearly()
    test_linked_cookbook_hosted_api_link_fails_clearly()
    test_linked_solver_selection_generated_html_link_fails_clearly()
    test_percent_encoded_generated_html_link_fails_after_resolution()
    test_backslash_escaped_generated_html_link_fails_after_resolution()
    test_root_relative_generated_html_link_fails_after_resolution()
    test_escaping_local_generated_html_link_fails_clearly()
    test_missing_route_fragment_fails_clearly()
    test_route_fragment_in_fenced_heading_does_not_satisfy_contract()
    test_missing_local_only_text_fails_clearly()
    test_hidden_local_only_text_does_not_satisfy_contract()
    test_missing_maintainer_claim_boundary_text_fails_clearly()
    test_missing_makefile_routing_dependency_fails_clearly()
    test_missing_makefile_docs_check_dependency_fails_clearly()
    test_missing_makefile_local_only_dependency_fails_clearly()
    test_makefile_validate_routing_only_dependency_fails_clearly()
    test_makefile_routing_extra_dependency_name_fails_clearly()
    test_missing_makefile_routing_target_fails_clearly()
    test_missing_makefile_freshness_dependency_fails_clearly()
    test_parallel_docs_check_generation_before_coverage_wiring_fails_clearly()
    test_makefile_freshness_extra_dependency_name_fails_clearly()


if __name__ == "__main__":
    main()
