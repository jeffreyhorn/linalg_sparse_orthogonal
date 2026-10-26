#!/usr/bin/env python3
"""Fail-closed and benign workflow mutations for the combined API-docs guard."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from test_api_docs_local_only_guard import run_guard, write_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from api_docs_workflow_policy import PolicyViolation, check_direct_policy  # noqa: E402
from api_docs_workflow_policy_inputs import read_policy_inputs  # noqa: E402
from api_docs_workflow_reader import WorkflowReadError  # noqa: E402


def workflow(steps: str, *, env: str = "") -> str:
    return (
        "name: api-docs-mutation\n"
        "on: [push]\n"
        f"{env}"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        f"{steps}"
    )


UPLOAD = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
SAFE_UPLOAD = "      - uses: actions/upload-artifact@v4\n        with:\n          path: build/output\n"

# Each row is a complete workflow mutation and its independent expected class.
CASES = (
    (
        "added_matching_upload", "REJECT", "stages docs",
        workflow("      - run: cp -R docs artifact/\n" + UPLOAD),
    ),
    (
        "aliased_flow_with_path", "REJECT", "publishes docs",
        workflow(
            "      - {uses: actions/upload-artifact@v4, with: {path: *docs_path}}\n",
            env="env: {DOCS_PATH: &docs_path docs/}\n",
        ),
    ),
    (
        "moved_generated_docs", "REJECT", "generated API output",
        workflow("      - run: mv docs/api/html artifact/\n" + UPLOAD),
    ),
    (
        "malformed_flow", "FAIL_CLOSED", "parse:",
        workflow("      - uses: actions/upload-artifact@v4\n"
                 "        with: {path: [docs/}\n"),
    ),
    (
        "duplicate_path_key", "FAIL_CLOSED", "parse: duplicate key 'path'",
        workflow("      - uses: actions/upload-artifact@v4\n"
                 "        with:\n          path: build/output\n          path: docs/\n"),
    ),
    (
        "dynamic_upload_path", "FAIL_CLOSED", "dynamic",
        workflow("      - uses: actions/upload-artifact@v4\n"
                 "        with:\n          path: ${{ inputs.output }}\n"),
    ),
    (
        "dynamic_action_identity", "FAIL_CLOSED", "dynamic or unresolved action identity",
        workflow("      - uses: ${{ inputs.publisher }}\n"
                 "        with:\n          path: build/output\n"),
    ),
    (
        "dynamic_staging_source", "FAIL_CLOSED", "dynamic staging paths",
        workflow("      - run: cp -R $SOURCE artifact/\n" + UPLOAD),
    ),
    (
        "safe_added_upload", "ALLOW", "",
        workflow("      - run: cp -R docs artifact/\n" + SAFE_UPLOAD),
    ),
    (
        "shell_comment", "ALLOW", "",
        workflow("      - run: echo ready # docs/api/html\n" + SAFE_UPLOAD),
    ),
    (
        "quoted_anchor_text", "ALLOW", "",
        workflow("      - run: 'echo \"path: &docs docs/\"'\n" + SAFE_UPLOAD),
    ),
    (
        "unrelated_checkout", "ALLOW", "",
        workflow("      - uses: actions/checkout@v4\n"
                 "        with:\n          path: build/checkout\n" + SAFE_UPLOAD),
    ),
    (
        "separate_jobs", "ALLOW", "",
        "name: api-docs-mutation\non: [push]\njobs:\n"
        "  producer:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - run: cp -R docs artifact/\n"
        "  consumer:\n    runs-on: ubuntu-latest\n    steps:\n" + UPLOAD,
    ),
)


class WorkflowMutationTests(unittest.TestCase):
    def test_mutations_have_unique_names_and_expected_classes(self):
        names = [name for name, _, _, _ in CASES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual({"ALLOW", "REJECT", "FAIL_CLOSED"}, {kind for _, kind, _, _ in CASES})

    def test_combined_guard_mutation_matrix(self):
        for name, expected, diagnostic, content in CASES:
            with self.subTest(mutation=name, expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write_fixture(root)
                    (root / ".github/workflows/api-docs.yml").write_text(
                        content, encoding="utf-8"
                    )
                    structured_error = None
                    try:
                        check_direct_policy(read_policy_inputs(root))
                    except (WorkflowReadError, PolicyViolation) as exc:
                        structured_error = str(exc)
                    result = run_guard(root)
                    output = result.stdout + result.stderr
                    if expected == "ALLOW":
                        self.assertIsNone(structured_error)
                        self.assertEqual(0, result.returncode, output)
                    else:
                        self.assertIsNotNone(structured_error)
                        self.assertIn(diagnostic, structured_error)
                        self.assertNotEqual(0, result.returncode, output)
                        self.assertIn(diagnostic, output)
                        self.assertIn(".github/workflows/api-docs.yml", output)

    def test_unrelated_checkout_path_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            (root / ".github/workflows/api-docs.yml").write_text(
                workflow("      - uses: actions/checkout@v4\n"
                         "        with:\n          path: docs/\n" + SAFE_UPLOAD),
                encoding="utf-8",
            )
            check_direct_policy(read_policy_inputs(root))
            result = run_guard(root)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
