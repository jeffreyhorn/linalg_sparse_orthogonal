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


def run_git(root: Path, *args: str) -> None:
    result = run(["git", *args], root)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


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


def test_workflow_generated_api_root_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: test -d docs/api\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "generated API output root")


def test_workflow_rooted_generated_api_root_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: test -d /docs/api\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "generated API output root")


def test_workflow_relative_generated_api_root_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: test -d ./docs/api\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "generated API output root")


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


def test_workflow_windows_generated_api_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: windows-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: .\\docs\\api\\html\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_workflow_mixed_case_backslash_generated_api_reference_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: windows-latest\n"
            "    steps:\n"
            "      - run: Test-Path .\\Docs\\Api\\Html\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "generated API output root")


def test_workflow_broad_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_windows_broad_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: windows-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: .\\docs\\\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_uppercase_windows_broad_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: windows-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: .\\Docs\\Api\\Html\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_workflow_flow_mapping_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with: {name: generated-api-html, path: docs/}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_quoted_flow_mapping_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            '        with: {"name": "generated-api-html", "path": "docs/"}\n',
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_spaced_path_key_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          path : docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_quoted_spaced_path_key_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            '          "path" : docs/\n',
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_local_action_broad_docs_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: ./.github/actions/publish\n"
            "        with:\n"
            "          path: docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_quoted_local_action_broad_docs_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: './.github/actions/publish'\n"
            "        with:\n"
            "          path: docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_unknown_publisher_broad_docs_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: acme/action@v1\n"
            "        with:\n"
            "          path: docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_quoted_uses_key_broad_docs_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - \"uses\": acme/publish@v1\n"
            "        with:\n"
            "          path: docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_docker_action_broad_docs_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: docker://publisher\n"
            "        with:\n"
            "          path: docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_publish_dir_block_docs_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: peaceiris/actions-gh-pages@v4\n"
            "        with:\n"
            "          publish_dir: |\n"
            "            ./docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_release_asset_broad_docs_files_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: softprops/action-gh-release@v2\n"
            "        with:\n"
            "          files: docs/**\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_release_asset_path_broad_docs_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-release-asset@v1\n"
            "        with:\n"
            "          asset_path: docs/api-html.tgz\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_relative_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: ./docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_quoted_broad_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            '          path: "docs"\n',
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_quoted_relative_broad_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            '          path: "./docs"\n',
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_workspace_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: ${{ github.workspace }}/docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_dynamic_artifact_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: ${{ env.DOCS_PATH }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_embedded_expression_artifact_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: docs${{ env.SUFFIX }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_dynamic_flow_mapping_artifact_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with: {name: generated-api-html, path: ${{ env.DOCS_PATH }}}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_dynamic_block_artifact_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: |\n"
            "            ${{ env.DOCS_PATH }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_dynamic_chomped_block_artifact_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: |-\n"
            "            ${{ env.DOCS_PATH }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_quoted_dynamic_block_artifact_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: |\n"
            "            \"${{ env.DOCS_PATH }}\"\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_embedded_dynamic_block_artifact_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: |\n"
            "            prefix/${{ env.DOCS_PATH }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_dynamic_release_files_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: softprops/action-gh-release@v2\n"
            "        with:\n"
            "          files: ${{ env.DOCS_PATH }}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_dynamic_release_asset_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-release-asset@v1\n"
            "        with:\n"
            "          asset_path: $DOCS_PATH\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_dynamic_command_publication_path_fails_closed() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: aws s3 sync \"$DOCS_PATH\" s3://example-generated-api/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "dynamic publication paths")


def test_workflow_unrelated_publisher_build_path_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make test\n"
            "      - run: aws s3 sync build/ s3://example-bucket/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_broad_workspace_docs_without_trailing_slash_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: ${{ github.workspace }}/docs\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_workspace_root_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: ${{ github.workspace }}/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_docs_glob_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: docs/**\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_root_glob_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: '**'\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_root_deep_glob_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: '**/*'\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_docs_single_glob_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: docs/*\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_docs_artifact_path_with_inline_comment_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: docs/ # publish docs\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_block_docs_artifact_path_with_inline_comment_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: |\n"
            "            ./docs/ # publish docs\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_repo_glob_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: ./**\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_normalized_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: ./build/../docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_docs_dot_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: docs/.\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_block_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: |\n"
            "            ./docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_chomped_block_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: >-\n"
            "            ./docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_broad_docs_custom_deploy_command_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: aws s3 sync docs/ s3://example-generated-api/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_folded_docs_publish_command_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: >\n"
            "          aws s3\n"
            "          sync docs/ s3://example-generated-api/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_rclone_docs_publish_command_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: rclone copy docs/ remote:generated-api\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_gh_release_upload_docs_glob_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: gh release upload \"$TAG\" docs/**\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_gh_pages_publish_dir_docs_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: peaceiris/actions-gh-pages@v4\n"
            "        with:\n"
            "          publish_dir: ./docs\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_staged_docs_artifact_upload_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: cp -R docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_moved_docs_artifact_upload_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: mv docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_archived_docs_artifact_upload_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: tar -czf artifact.tgz docs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact.tgz\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "archives docs for publication")


def test_workflow_7z_archived_docs_artifact_upload_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: 7z a artifact.7z docs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact.7z\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "archives docs for publication")


def test_tracked_generated_api_file_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        run_git(root, "config", "user.email", "test@example.com")
        run_git(root, "config", "user.name", "Test User")
        run_git(root, "add", "-f", "docs/api/html/index.html")
        run_git(root, "commit", "-qm", "track generated api")

    assert_guard_fails_with(mutate, "tracked; local-only generated HTML")


def test_staged_generated_api_file_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / "docs" / "api" / "html" / "staged.html").write_text(
            "<!doctype html>\n",
            encoding="utf-8",
        )
        run_git(root, "add", "-f", "docs/api/html/staged.html")

    assert_guard_fails_with(mutate, "staged; unstage them")


def test_visible_untracked_generated_api_file_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".gitignore").write_text("", encoding="utf-8")

    assert_guard_fails_with(mutate, "visible as non-ignored untracked files")


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
    test_workflow_generated_api_root_path_fails_clearly()
    test_workflow_rooted_generated_api_root_path_fails_clearly()
    test_workflow_relative_generated_api_root_path_fails_clearly()
    test_workflow_publication_semantics_fail_clearly()
    test_workflow_windows_generated_api_artifact_path_fails_clearly()
    test_workflow_mixed_case_backslash_generated_api_reference_fails_clearly()
    test_workflow_broad_docs_artifact_path_fails_clearly()
    test_workflow_windows_broad_docs_artifact_path_fails_clearly()
    test_workflow_uppercase_windows_broad_docs_artifact_path_fails_clearly()
    test_workflow_flow_mapping_docs_artifact_path_fails_clearly()
    test_workflow_quoted_flow_mapping_docs_artifact_path_fails_clearly()
    test_workflow_spaced_path_key_docs_artifact_path_fails_clearly()
    test_workflow_quoted_spaced_path_key_docs_artifact_path_fails_clearly()
    test_workflow_local_action_broad_docs_path_fails_closed()
    test_workflow_quoted_local_action_broad_docs_path_fails_closed()
    test_workflow_unknown_publisher_broad_docs_path_fails_closed()
    test_workflow_quoted_uses_key_broad_docs_path_fails_closed()
    test_workflow_docker_action_broad_docs_path_fails_closed()
    test_workflow_publish_dir_block_docs_path_fails_clearly()
    test_workflow_release_asset_broad_docs_files_fails_closed()
    test_workflow_release_asset_path_broad_docs_fails_closed()
    test_workflow_broad_relative_docs_artifact_path_fails_clearly()
    test_workflow_quoted_broad_docs_artifact_path_fails_clearly()
    test_workflow_quoted_relative_broad_docs_artifact_path_fails_clearly()
    test_workflow_broad_workspace_docs_artifact_path_fails_clearly()
    test_workflow_dynamic_artifact_path_fails_closed()
    test_workflow_embedded_expression_artifact_path_fails_closed()
    test_workflow_dynamic_flow_mapping_artifact_path_fails_closed()
    test_workflow_dynamic_block_artifact_path_fails_closed()
    test_workflow_dynamic_chomped_block_artifact_path_fails_closed()
    test_workflow_quoted_dynamic_block_artifact_path_fails_closed()
    test_workflow_embedded_dynamic_block_artifact_path_fails_closed()
    test_workflow_dynamic_release_files_path_fails_closed()
    test_workflow_dynamic_release_asset_path_fails_closed()
    test_workflow_dynamic_command_publication_path_fails_closed()
    test_workflow_unrelated_publisher_build_path_is_allowed()
    test_workflow_broad_workspace_docs_without_trailing_slash_fails_clearly()
    test_workflow_broad_workspace_root_artifact_path_fails_clearly()
    test_workflow_broad_docs_glob_artifact_path_fails_clearly()
    test_workflow_broad_root_glob_artifact_path_fails_clearly()
    test_workflow_broad_root_deep_glob_artifact_path_fails_clearly()
    test_workflow_broad_docs_single_glob_artifact_path_fails_clearly()
    test_workflow_broad_docs_artifact_path_with_inline_comment_fails_clearly()
    test_workflow_broad_block_docs_artifact_path_with_inline_comment_fails_clearly()
    test_workflow_broad_repo_glob_artifact_path_fails_clearly()
    test_workflow_broad_normalized_docs_artifact_path_fails_clearly()
    test_workflow_broad_docs_dot_artifact_path_fails_clearly()
    test_workflow_broad_block_docs_artifact_path_fails_clearly()
    test_workflow_broad_chomped_block_docs_artifact_path_fails_clearly()
    test_workflow_broad_docs_custom_deploy_command_fails_clearly()
    test_workflow_folded_docs_publish_command_fails_clearly()
    test_workflow_rclone_docs_publish_command_fails_clearly()
    test_workflow_gh_release_upload_docs_glob_fails_clearly()
    test_workflow_gh_pages_publish_dir_docs_fails_clearly()
    test_workflow_staged_docs_artifact_upload_fails_clearly()
    test_workflow_moved_docs_artifact_upload_fails_clearly()
    test_workflow_archived_docs_artifact_upload_fails_clearly()
    test_workflow_7z_archived_docs_artifact_upload_fails_clearly()
    test_tracked_generated_api_file_fails_clearly()
    test_staged_generated_api_file_fails_clearly()
    test_visible_untracked_generated_api_file_fails_clearly()
    test_missing_local_only_wording_fails_clearly()


if __name__ == "__main__":
    main()
