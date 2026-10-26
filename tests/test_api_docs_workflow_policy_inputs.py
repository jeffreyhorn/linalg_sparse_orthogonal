#!/usr/bin/env python3
"""Contract checks for the structured workflow policy handoff."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from api_docs_workflow_policy_inputs import read_policy_inputs  # noqa: E402
from test_api_docs_local_only_guard import run_guard, write_fixture  # noqa: E402


class PolicyInputTests(unittest.TestCase):
    def test_working_directory_precedence_reaches_run_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "case.yml").write_text(
                "defaults:\n  run:\n    working-directory: workflow-dir\n"
                "jobs:\n  first:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - run: echo workflow\n"
                "  second:\n    runs-on: ubuntu-latest\n"
                "    defaults:\n      run:\n        working-directory: job-dir\n"
                "    steps:\n      - run: echo job\n"
                "      - run: echo step\n        working-directory: step-dir\n",
                encoding="utf-8",
            )
            inputs = read_policy_inputs(root)
            self.assertEqual(
                [item.working_directory.value for item in inputs],
                ["workflow-dir", "job-dir", "step-dir"],
            )

    def test_real_workflow_inventory(self):
        inputs = read_policy_inputs(REPO_ROOT)
        self.assertEqual(len(inputs), 110)
        self.assertEqual(
            {(item.workflow_path.name, item.job_key) for item in inputs if item.step_index is None},
            set(),
        )
        uploads = [
            item for item in inputs
            if item.action_identity is not None
            and item.action_identity.value == "actions/upload-artifact@v4"
        ]
        self.assertEqual(len(uploads), 9)
        self.assertTrue(all(len(item.path_inputs) == 1 for item in uploads))
        self.assertTrue(all(item.path_inputs[0].entries for item in uploads))
        self.assertTrue(all(
            entry.canonical is not None
            for item in uploads for path in item.path_inputs for entry in path.entries
        ))
        self.assertEqual(
            {item.shell_family for item in inputs if item.rendered_command is not None},
            {"posix", "powershell", "cmd"},
        )

    def test_path_canonicalization_does_not_rewrite_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "case.yaml").write_text(
                "name: case\non: [push]\njobs:\n  build:\n    runs-on: windows-latest\n"
                "    steps:\n      - run: |\n          Copy-Item -Path .\\\\docs\\\\* `\n"
                "            -Destination artifact\n        shell: pwsh\n"
                "      - uses: actions/upload-artifact@v4\n        with:\n"
                "          path: .\\\\BUILD\\\\..\\\\Docs\\\\*\n",
                encoding="utf-8",
            )
            run, upload = read_policy_inputs(root)
            self.assertIn(".\\\\docs\\\\*", run.rendered_command.value)
            self.assertEqual(run.shell_family, "powershell")
            self.assertEqual(upload.path_inputs[0].canonical, "docs/*")
            self.assertEqual(upload.path_inputs[0].raw.location.path.name, "case.yaml")
            self.assertEqual(upload.step_index, 1)
            self.assertEqual(upload.job_key, "build")

    def test_dynamic_path_is_not_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "case.yml").write_text(
                "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: actions/upload-artifact@v4\n"
                "        with:\n          path: ${{ env.ARTIFACT }}\n",
                encoding="utf-8",
            )
            item = read_policy_inputs(root)[0]
            self.assertTrue(item.path_inputs[0].dynamic)
            self.assertIsNone(item.path_inputs[0].canonical)

    def test_multiline_paths_keep_each_literal_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "case.yml").write_text(
                "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: actions/upload-artifact@v4\n"
                "        with:\n          path: |\n"
                "            ../DoCs/**\n            ./build/../coverage/*.html\n"
                "            ${{ env.OUTPUT }}\n",
                encoding="utf-8",
            )
            entries = read_policy_inputs(root)[0].path_inputs[0].entries
            self.assertEqual(
                [entry.canonical for entry in entries],
                ["../docs/**", "coverage/*.html", None],
            )
            self.assertEqual([entry.dynamic for entry in entries], [False, False, True])
            self.assertEqual([entry.glob for entry in entries], [True, True, False])

    def test_dynamic_action_identity_stays_unresolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "case.yml").write_text(
                "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: ${{ env.ACTION }}\n",
                encoding="utf-8",
            )
            item = read_policy_inputs(root)[0]
            self.assertTrue(item.action_dynamic)
            self.assertEqual(item.action_identity.location.line, 5)

    def test_staging_requires_an_explicit_link_between_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "case.yml").write_text(
                "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - run: cp docs/index.html artifact/index.html\n"
                "      - uses: actions/upload-artifact@v4\n"
                "        with:\n          path: artifact/\n",
                encoding="utf-8",
            )
            producer, consumer = read_policy_inputs(root)
            self.assertEqual((producer.step_index, consumer.step_index), (0, 1))
            self.assertEqual(producer.rendered_command.value, "cp docs/index.html artifact/index.html")
            self.assertIsNone(consumer.rendered_command)
            self.assertEqual(consumer.path_inputs[0].canonical, "artifact")
            self.assertEqual(producer.staging_relations, ())
            self.assertEqual(consumer.staging_relations, ())

    def test_guard_rejects_malformed_workflow_at_parser_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            (root / ".github" / "workflows" / "ci.yml").write_text(
                "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - run: echo ready\n        run: echo hidden\n",
                encoding="utf-8",
            )
            result = run_guard(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("parse: duplicate key 'run'", result.stderr)
            self.assertIn("ci.yml:6:", result.stderr)
            self.assertIn("structured workflow publication policy failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
