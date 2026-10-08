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


def test_workflow_quoted_hash_before_generated_api_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'printf \"marker # value\"; ls docs/api/html'\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "generated API HTML output path")


def test_workflow_escaped_run_generated_api_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"ls \\x64ocs/api/html\"\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "generated API HTML output path")


def test_workflow_plain_scalar_apostrophe_before_comment_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: Don't publish # docs/api/html remains local-only\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make test\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_multiline_single_quoted_trailing_comment_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: 'Build followed by\n"
            "  output' # docs/api/html remains local-only\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make test\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_multiline_double_quoted_trailing_comment_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: \"Build followed by\n"
            "  output\" # docs/api/html remains local-only\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make test\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_single_quoted_doubled_apostrophe_before_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'printf ''marker # value''; cp -R docs/api/html artifact/'\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_yaml_workflow_generated_api_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yaml").write_text(
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


def test_workflow_escaped_generated_api_artifact_path_fails_clearly() -> None:
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
            "          path: \"\\x64ocs/api/html\"\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_workflow_escaped_upload_artifact_action_reference_fails() -> None:
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
            "      - uses: \"actions/upl\\x6fad-art\\x69fact@v4\"\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_flow_mapping_escaped_generated_api_artifact_path_fails() -> None:
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
            "        with: {name: generated-api-html, path: \"\\x64ocs/api/html\"}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_workflow_flow_mapping_multiline_escaped_generated_api_artifact_path_fails() -> None:
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
            "        with: {name: generated-api-html, path: \"\\x64ocs/\\\n"
            "            api/html\"}\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_workflow_multiline_escaped_generated_api_artifact_path_fails() -> None:
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
            "          path: \"\\x64ocs/\\\n"
            "            api/html\"\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_workflow_newline_escaped_docs_artifact_path_fails() -> None:
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
            "          path: \"build/\\ndocs/\"\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


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


def test_workflow_anchored_docs_artifact_path_fails_clearly() -> None:
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
            "          path: &docs docs/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_aliased_docs_artifact_path_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "env:\n"
            "  DOCS_PATH: &docs docs/\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: *docs\n",
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


def test_workflow_non_publication_action_docs_path_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/checkout@v4\n"
            "        with:\n"
            "          path: docs/tutorial.md\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_aliased_build_artifact_path_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "env:\n"
            "  BUILD_PATH: &build build/\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: mkdir -p build && printf ok > build/result.txt\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: *build\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_non_publication_action_dynamic_path_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - uses: actions/checkout@v4\n"
            "        with:\n"
            "          path: ${{ env.CHECKOUT_PATH }}\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


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


def test_workflow_unrelated_publisher_with_unrelated_variable_is_allowed() -> None:
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
            "      - run: echo \"$TAG\"\n"
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


def test_workflow_quoted_hash_before_docs_publish_command_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: make api-docs-freshness\n"
            "      - run: 'printf \"marker # value\"; aws s3 sync docs/ s3://example-generated-api/'\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publishes docs or repository roots")


def test_workflow_literal_run_hash_before_generated_copy_fails_clearly() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          printf \"marker # value\"; cp -R docs/api/html artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


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


def test_workflow_folded_staged_docs_artifact_upload_fails_clearly() -> None:
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
            "          cp -R\n"
            "          docs artifact/\n"
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


def test_workflow_folded_archived_docs_artifact_upload_fails_clearly() -> None:
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
            "          tar -czf artifact.tgz\n"
            "          docs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact.tgz\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "archives docs for publication")


def test_workflow_literal_continued_docs_copy_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          cp -R \\\n"
            "            docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_literal_comment_backslash_before_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          # preparation \\\n"
            "          cp -R \\\n"
            "          docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_literal_continued_quoted_hash_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          printf \"marker \\\n"
            "          # value\"; cp -R \\\n"
            "          docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_literal_continued_docs_archive_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          tar -czf artifact.tgz \\\n"
            "            docs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact.tgz\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "archives docs for publication")


def test_workflow_literal_split_token_docs_copy_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          cp -R do\\\n"
            "          cs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_literal_split_token_docs_archive_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          tar -czf artifact.tgz do\\\n"
            "          cs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact.tgz\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "archives docs for publication")


def test_workflow_multiline_plain_docs_copy_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: cp -R\n"
            "          docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_multiline_flow_run_docs_copy_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - {run: cp -R\n"
            "          docs artifact/}\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_multiline_named_flow_run_docs_copy_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - {name: Stage, run: cp -R\n"
            "          docs artifact/}\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_multiline_plain_docs_archive_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: tar -czf artifact.tgz\n"
            "          docs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact.tgz\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "archives docs for publication")


def test_workflow_multiline_quoted_docs_copy_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"cp -R\n"
            "          docs artifact/\"\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_multiline_single_quoted_docs_copy_with_trailing_comment_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'cp -R\n"
            "          docs artifact/' # stage docs\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_multiline_double_quoted_docs_copy_with_trailing_comment_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"cp -R\n"
            "          docs artifact/\" # stage docs\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_multiline_quoted_docs_archive_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'tar -czf artifact.tgz\n"
            "          docs/'\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact.tgz\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "archives docs for publication")


def test_workflow_multiline_single_quoted_hash_before_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'printf \"marker\n"
            "          # value\"; cp -R docs artifact/'\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_multiline_double_quoted_hash_before_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"printf \\\"marker\n"
            "          # value\\\"; cp -R docs artifact/\"\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_double_quoted_yaml_escaped_hash_before_split_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"printf \\\"marker # value\\\"; cp -R\n"
            "          docs artifact/\"\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_double_quoted_yaml_newline_comment_before_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"echo marker # note\\ncp -R\n"
            "          docs artifact/\"\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_double_quoted_yaml_escaped_linebreak_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"cp -R do\\\n"
            "          cs artifact/\"\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_double_quoted_yaml_escaped_linebreak_docs_archive_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"tar -czf artifact.tgz do\\\n"
            "          cs/\"\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact.tgz\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "archives docs for publication")


def test_workflow_double_quoted_yaml_hex_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"cp -R \\x64ocs artifact/\"\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_literal_shell_multiline_string_before_generated_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          printf \"marker\n"
            "          # value\"; cp -R docs/api/html artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "publication, artifact, or Pages semantics")


def test_workflow_multiline_quoted_hash_before_split_docs_copy_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'printf \"marker\n"
            "          # value\"; cp -R\n"
            "          docs artifact/'\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_multiline_quoted_split_docs_copy_blank_echo_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'cp -R\n"
            "          docs artifact/\n"
            "\n"
            "          echo complete'\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_quoted_run_key_folded_docs_copy_artifact_upload_fails() -> None:
    def mutate(root: Path) -> None:
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: generated-api\n"
            "on: [push]\n"
            "jobs:\n"
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - \"run\": >\n"
            "          cp -R\n"
            "          docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: generated-api-html\n"
            "          path: artifact/\n",
            encoding="utf-8",
        )

    assert_guard_fails_with(mutate, "stages docs for publication")


def test_workflow_unrelated_build_copy_before_check_docs_upload_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: cp build/result.txt build/copy.txt\n"
            "      - name: Check docs\n"
            "        run: make docs-check\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_unrelated_build_archive_before_check_docs_upload_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: tar -czf build/results.tgz build/results/\n"
            "      - name: Check docs\n"
            "        run: make docs-check\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_folded_build_copy_before_same_step_name_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: >\n"
            "          cp build/result.txt build/copy.txt\n"
            "        name: Check docs\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_literal_block_before_commented_sibling_name_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          make test\n"
            "        name: Check docs # docs/api/html remains local-only\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_literal_shell_comment_before_build_copy_upload_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          # docs/api/html remains local-only\n"
            "          cp build/result.txt build/copy.txt\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_literal_build_copy_then_echo_docs_upload_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          cp build/result.txt build/copy.txt\n"
            "          echo docs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_multiline_quoted_build_copy_blank_echo_docs_upload_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'cp build/result.txt build/copy.txt\n"
            "\n"
            "          echo docs/'\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_multiline_quoted_build_archive_blank_echo_docs_upload_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: \"tar -czf build/results.tgz build/results/\n"
            "\n"
            "          echo docs/\"\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_folded_build_copy_then_indented_echo_docs_upload_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: >\n"
            "          cp build/result.txt build/copy.txt\n"
            "            echo docs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


def test_workflow_literal_build_archive_then_echo_docs_upload_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        write_fixture(root)
        (root / ".github" / "workflows" / "api-docs.yml").write_text(
            "name: build-artifact\n"
            "on: [push]\n"
            "jobs:\n"
            "  package:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          tar -czf build/results.tgz build/results/\n"
            "          echo docs/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: build-output\n"
            "          path: build/\n",
            encoding="utf-8",
        )
        result = run_guard(root)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)


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
    test_workflow_quoted_hash_before_generated_api_path_fails_clearly()
    test_workflow_escaped_run_generated_api_path_fails_clearly()
    test_workflow_plain_scalar_apostrophe_before_comment_is_allowed()
    test_workflow_multiline_single_quoted_trailing_comment_is_allowed()
    test_workflow_multiline_double_quoted_trailing_comment_is_allowed()
    test_workflow_single_quoted_doubled_apostrophe_before_copy_fails()
    test_yaml_workflow_generated_api_path_fails_clearly()
    test_workflow_generated_api_root_path_fails_clearly()
    test_workflow_rooted_generated_api_root_path_fails_clearly()
    test_workflow_relative_generated_api_root_path_fails_clearly()
    test_workflow_publication_semantics_fail_clearly()
    test_workflow_windows_generated_api_artifact_path_fails_clearly()
    test_workflow_escaped_generated_api_artifact_path_fails_clearly()
    test_workflow_escaped_upload_artifact_action_reference_fails()
    test_workflow_flow_mapping_escaped_generated_api_artifact_path_fails()
    test_workflow_flow_mapping_multiline_escaped_generated_api_artifact_path_fails()
    test_workflow_multiline_escaped_generated_api_artifact_path_fails()
    test_workflow_newline_escaped_docs_artifact_path_fails()
    test_workflow_mixed_case_backslash_generated_api_reference_fails_clearly()
    test_workflow_broad_docs_artifact_path_fails_clearly()
    test_workflow_anchored_docs_artifact_path_fails_clearly()
    test_workflow_aliased_docs_artifact_path_fails_clearly()
    test_workflow_windows_broad_docs_artifact_path_fails_clearly()
    test_workflow_uppercase_windows_broad_docs_artifact_path_fails_clearly()
    test_workflow_flow_mapping_docs_artifact_path_fails_clearly()
    test_workflow_quoted_flow_mapping_docs_artifact_path_fails_clearly()
    test_workflow_spaced_path_key_docs_artifact_path_fails_clearly()
    test_workflow_quoted_spaced_path_key_docs_artifact_path_fails_clearly()
    test_workflow_local_action_broad_docs_path_fails_closed()
    test_workflow_quoted_local_action_broad_docs_path_fails_closed()
    test_workflow_non_publication_action_docs_path_is_allowed()
    test_workflow_aliased_build_artifact_path_is_allowed()
    test_workflow_non_publication_action_dynamic_path_is_allowed()
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
    test_workflow_unrelated_publisher_with_unrelated_variable_is_allowed()
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
    test_workflow_quoted_hash_before_docs_publish_command_fails_clearly()
    test_workflow_literal_run_hash_before_generated_copy_fails_clearly()
    test_workflow_folded_docs_publish_command_fails_clearly()
    test_workflow_rclone_docs_publish_command_fails_clearly()
    test_workflow_gh_release_upload_docs_glob_fails_clearly()
    test_workflow_gh_pages_publish_dir_docs_fails_clearly()
    test_workflow_staged_docs_artifact_upload_fails_clearly()
    test_workflow_moved_docs_artifact_upload_fails_clearly()
    test_workflow_folded_staged_docs_artifact_upload_fails_clearly()
    test_workflow_archived_docs_artifact_upload_fails_clearly()
    test_workflow_folded_archived_docs_artifact_upload_fails_clearly()
    test_workflow_literal_continued_docs_copy_artifact_upload_fails()
    test_workflow_literal_comment_backslash_before_docs_copy_fails()
    test_workflow_literal_continued_quoted_hash_docs_copy_fails()
    test_workflow_literal_continued_docs_archive_artifact_upload_fails()
    test_workflow_literal_split_token_docs_copy_artifact_upload_fails()
    test_workflow_literal_split_token_docs_archive_artifact_upload_fails()
    test_workflow_multiline_plain_docs_copy_artifact_upload_fails()
    test_workflow_multiline_flow_run_docs_copy_artifact_upload_fails()
    test_workflow_multiline_named_flow_run_docs_copy_artifact_upload_fails()
    test_workflow_multiline_plain_docs_archive_artifact_upload_fails()
    test_workflow_multiline_quoted_docs_copy_artifact_upload_fails()
    test_workflow_multiline_single_quoted_docs_copy_with_trailing_comment_fails()
    test_workflow_multiline_double_quoted_docs_copy_with_trailing_comment_fails()
    test_workflow_multiline_quoted_docs_archive_artifact_upload_fails()
    test_workflow_multiline_single_quoted_hash_before_docs_copy_fails()
    test_workflow_multiline_double_quoted_hash_before_docs_copy_fails()
    test_workflow_double_quoted_yaml_escaped_hash_before_split_docs_copy_fails()
    test_workflow_double_quoted_yaml_newline_comment_before_docs_copy_fails()
    test_workflow_double_quoted_yaml_escaped_linebreak_docs_copy_fails()
    test_workflow_double_quoted_yaml_escaped_linebreak_docs_archive_fails()
    test_workflow_double_quoted_yaml_hex_docs_copy_fails()
    test_workflow_literal_shell_multiline_string_before_generated_copy_fails()
    test_workflow_multiline_quoted_hash_before_split_docs_copy_fails()
    test_workflow_multiline_quoted_split_docs_copy_blank_echo_fails()
    test_workflow_quoted_run_key_folded_docs_copy_artifact_upload_fails()
    test_workflow_unrelated_build_copy_before_check_docs_upload_is_allowed()
    test_workflow_unrelated_build_archive_before_check_docs_upload_is_allowed()
    test_workflow_folded_build_copy_before_same_step_name_is_allowed()
    test_workflow_literal_block_before_commented_sibling_name_is_allowed()
    test_workflow_literal_shell_comment_before_build_copy_upload_is_allowed()
    test_workflow_literal_build_copy_then_echo_docs_upload_is_allowed()
    test_workflow_multiline_quoted_build_copy_blank_echo_docs_upload_is_allowed()
    test_workflow_multiline_quoted_build_archive_blank_echo_docs_upload_is_allowed()
    test_workflow_folded_build_copy_then_indented_echo_docs_upload_is_allowed()
    test_workflow_literal_build_archive_then_echo_docs_upload_is_allowed()
    test_workflow_7z_archived_docs_artifact_upload_fails_clearly()
    test_tracked_generated_api_file_fails_clearly()
    test_staged_generated_api_file_fails_clearly()
    test_visible_untracked_generated_api_file_fails_clearly()
    test_missing_local_only_wording_fails_clearly()


if __name__ == "__main__":
    main()
