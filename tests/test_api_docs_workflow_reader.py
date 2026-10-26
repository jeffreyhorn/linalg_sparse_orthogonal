#!/usr/bin/env python3
"""Parser-level checks for the structured API-docs workflow reader."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from api_docs_workflow_reader import (  # noqa: E402
    WorkflowReadError,
    read_workflow,
    read_workflows,
)


FIXTURES = REPO_ROOT / "tests" / "fixtures" / "api_docs_yaml_edges"
MANIFEST = (
    REPO_ROOT
    / "docs"
    / "planning"
    / "EPIC_20"
    / "SPRINT_220"
    / "artifacts"
    / "day3-edge-fixture-matrix.tsv"
)


class WorkflowReaderTests(unittest.TestCase):
    def parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.yaml"
            path.write_text(text, encoding="utf-8")
            return read_workflow(path)

    def test_real_workflows_parse_into_bounded_jobs_and_steps(self):
        records = read_workflows(REPO_ROOT)
        self.assertEqual(
            {record.path.name for record in records},
            {"ci.yml", "macos-ci.yml", "windows-ci.yml"},
        )
        self.assertGreater(sum(len(job.steps) for record in records for job in record.jobs), 0)
        upload_steps = [
            step
            for record in records
            for job in record.jobs
            for step in job.steps
            if step.action_identity is not None
            and step.action_identity.value == "actions/upload-artifact@v4"
        ]
        self.assertEqual(len(upload_steps), 9)
        self.assertTrue(all("path" in dict(step.with_inputs) for step in upload_steps))
        self.assertTrue(
            any(
                value.tag == "tag:yaml.org,2002:int"
                for step in upload_steps
                for name, value in step.with_inputs
                if name == "retention-days"
            )
        )
        for record in records:
            self.assertGreater(len(record.jobs), 0)
            for job in record.jobs:
                for index, step in enumerate(job.steps):
                    self.assertEqual(step.workflow_path, record.path)
                    self.assertEqual(step.job_key, job.key)
                    self.assertEqual(step.step_index, index)
                    self.assertEqual(step.action_identity is None, step.run_value is not None)
                    self.assertGreater(step.location.line, 0)

    def test_frozen_edge_matrix_parser_stages(self):
        with MANIFEST.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {row["fixture"] for row in rows},
            {path.name for path in FIXTURES.glob("*.yml")},
        )
        for row in rows:
            with self.subTest(fixture=row["fixture"]):
                path = FIXTURES / row["fixture"]
                if row["failure_stage"] in {"parse", "unsupported"}:
                    with self.assertRaises(WorkflowReadError) as error:
                        read_workflow(path)
                    self.assertEqual(error.exception.kind, row["failure_stage"])
                    self.assertIn(str(path), str(error.exception))
                    self.assertGreater(error.exception.line, 0)
                else:
                    record = read_workflow(path)
                    self.assertEqual(len(record.jobs), 1)
                    self.assertGreater(len(record.jobs[0].steps), 0)

    def test_scalar_styles_source_spans_and_shell_precedence(self):
        record = self.parse_text(
            "name: scalar checks\n"
            "on: [push]\n"
            "defaults: {run: {shell: bash}}\n"
            "jobs:\n"
            "  windows:\n"
            "    runs-on: windows-2022\n"
            "    defaults: {run: {shell: pwsh}}\n"
            "    steps:\n"
            "      - run: >-\n"
            "          echo ready\n"
            "          echo done\n"
            "      - run: |\n"
            "          Write-Host ready\n"
            "        shell: cmd\n"
            "  linux:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps: [{run: 'echo safe # literal'}]\n"
        )
        windows = record.jobs[0].steps
        self.assertEqual(windows[0].effective_shell, "pwsh")
        self.assertEqual(windows[0].shell_source, "job")
        self.assertEqual(windows[0].shell_family, "powershell")
        self.assertEqual(windows[0].run_value.value, "echo ready echo done")
        self.assertEqual(windows[0].run_value.style, ">")
        self.assertTrue(windows[0].run_value.source.startswith(">-"))
        self.assertEqual(windows[1].effective_shell, "cmd")
        self.assertEqual(windows[1].shell_source, "step")
        self.assertEqual(windows[1].shell_family, "cmd")
        self.assertEqual(windows[1].run_value.style, "|")
        self.assertEqual(windows[1].run_value.value, "Write-Host ready\n")
        self.assertEqual(record.jobs[1].steps[0].effective_shell, "bash")
        self.assertEqual(record.jobs[1].steps[0].shell_source, "workflow")
        self.assertEqual(record.jobs[1].steps[0].run_value.value, "echo safe # literal")

    def test_windows_runner_default_and_dynamic_runner_unknown(self):
        record = self.parse_text(
            "jobs:\n"
            "  win: {runs-on: windows-2022, steps: [{run: 'Write-Host hi'}]}\n"
            "  dynamic: {runs-on: '${{ matrix.os }}', steps: [{run: echo hi}]}\n"
        )
        self.assertEqual(record.jobs[0].steps[0].effective_shell, "pwsh")
        self.assertEqual(record.jobs[0].steps[0].shell_source, "runner")
        self.assertEqual(record.jobs[0].steps[0].shell_family, "powershell")
        self.assertEqual(record.jobs[1].steps[0].effective_shell, "unknown")
        self.assertEqual(record.jobs[1].steps[0].shell_family, "unknown")

    def test_job_alias_is_represented_without_step_splicing(self):
        record = self.parse_text(
            "jobs:\n"
            "  first: &job\n"
            "    runs-on: ubuntu-latest\n"
            "    steps: [{run: echo ok}]\n"
            "  second: *job\n"
        )
        self.assertEqual([job.key for job in record.jobs], ["first", "second"])
        self.assertEqual([job.steps[0].job_key for job in record.jobs], ["first", "second"])
        self.assertEqual(record.jobs[1].location.line, 5)
        self.assertEqual(record.jobs[1].location.definition_line, 2)

    def test_folded_comment_literal_comment_and_chomping(self):
        record = self.parse_text(
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: >\n"
            "          echo ready # explanation\n"
            "          docs/api/html remains local-only\n"
            "      - run: |-\n"
            "          echo ready # explanation\n"
            "          docs/api/html remains local-only\n"
            "      - run: >+\n"
            "          echo ready\n"
            "          echo done\n"
            "\n"
            "      - run: >\n"
            "          cp build/result.txt build/copy.txt\n"
            "            echo docs/\n"
        )
        steps = record.jobs[0].steps
        self.assertEqual(
            steps[0].run_value.value,
            "echo ready # explanation docs/api/html remains local-only\n",
        )
        self.assertEqual(
            steps[1].run_value.value,
            "echo ready # explanation\ndocs/api/html remains local-only",
        )
        self.assertEqual(steps[2].run_value.value, "echo ready echo done\n\n")
        self.assertEqual(
            steps[3].run_value.value,
            "cp build/result.txt build/copy.txt\n  echo docs/\n",
        )

    def test_quoted_lines_escapes_and_yaml_comments(self):
        record = self.parse_text(
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: 'echo ok # harmless\n"
            "          docs/api/html remains local-only'\n"
            "      - run: \"echo ok\n"
            "          # docs/api/html remains local-only\"\n"
            "      - run: \"echo \\x64ocs\"\n"
            "      - run: \"echo safe\" # docs/api/html remains local-only\n"
        )
        values = [step.run_value.value for step in record.jobs[0].steps]
        self.assertEqual(values[0], "echo ok # harmless docs/api/html remains local-only")
        self.assertEqual(values[1], "echo ok # docs/api/html remains local-only")
        self.assertEqual(values[2], "echo docs")
        self.assertEqual(values[3], "echo safe")

    def test_scalar_alias_marks_use_and_definition(self):
        record = self.parse_text(
            "env:\n"
            "  STAGE: &stage >-\n"
            "    cp -R\n"
            "    docs artifact/\n"
            "  DOCS_PATH: &docs docs/\n"
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: *stage\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with: {path: *docs}\n"
        )
        run_step, upload_step = record.jobs[0].steps
        self.assertEqual(run_step.run_value.value, "cp -R docs artifact/")
        self.assertEqual(run_step.run_value.source, "*stage")
        self.assertEqual(run_step.run_value.location.line, 10)
        self.assertEqual(run_step.run_value.location.definition_line, 2)
        path = dict(upload_step.with_inputs)["path"]
        self.assertEqual(path.value, "docs/")
        self.assertEqual(path.source, "*docs")
        self.assertEqual(path.location.line, 12)
        self.assertEqual(path.location.definition_line, 5)

    def test_flow_steps_and_mapping_alias_do_not_hide_upload_path(self):
        record = self.parse_text(
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with: &inputs {path: docs/api/html, name: generated-api}\n"
            "      - {uses: actions/upload-artifact@v4, with: *inputs}\n"
        )
        first, second = record.jobs[0].steps
        self.assertEqual(dict(first.with_inputs)["path"].value, "docs/api/html")
        self.assertEqual(dict(second.with_inputs)["path"].value, "docs/api/html")
        self.assertEqual(second.with_location.line, 7)
        self.assertEqual(second.with_location.definition_line, 6)
        self.assertEqual(second.step_index, 1)

    def test_aliased_publishing_step_retains_uses_and_path(self):
        record = self.parse_text(
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - &publish {uses: actions/upload-artifact@v4, "
            "with: {path: docs/api/html}}\n"
            "      - *publish\n"
        )
        first, second = record.jobs[0].steps
        self.assertEqual(first.action_identity.value, "actions/upload-artifact@v4")
        self.assertEqual(second.action_identity.value, first.action_identity.value)
        self.assertEqual(dict(second.with_inputs)["path"].value, "docs/api/html")
        self.assertEqual(second.location.line, 6)
        self.assertEqual(second.location.definition_line, 5)
        self.assertEqual(second.step_index, 1)

    def test_powershell_backtick_and_windows_path_survive_shell_selection(self):
        record = self.parse_text(
            "jobs:\n"
            "  win:\n"
            "    runs-on: windows-latest\n"
            "    steps:\n"
            "      - run: |\n"
            "          Copy-Item -Recurse -Path `\n"
            "            .\\docs\\* -Destination artifact\n"
            "      - run: |\n"
            "          Copy-Item -Recurse -Path `\n"
            "            .\\docs\\* -Destination artifact\n"
            "        shell: bash\n"
            "      - run: |\n"
            "          Copy-Item -Recurse -Path `\n"
            "            .\\build\\* -Destination artifact\n"
            "        shell: powershell\n"
        )
        pwsh, bash, powershell = record.jobs[0].steps
        expected = "Copy-Item -Recurse -Path `\n  .\\docs\\* -Destination artifact\n"
        self.assertEqual(pwsh.run_value.value, expected)
        self.assertEqual(bash.run_value.value, expected)
        self.assertEqual(pwsh.shell_family, "powershell")
        self.assertEqual(bash.shell_family, "posix")
        self.assertEqual(powershell.shell_family, "powershell")
        self.assertIn(".\\build\\*", powershell.run_value.value)

    def test_yaml_double_quoted_windows_path_decodes_only_yaml_escapes(self):
        record = self.parse_text(
            "jobs:\n"
            "  win:\n"
            "    runs-on: windows-latest\n"
            "    steps:\n"
            '      - run: "Copy-Item -Path .\\\\docs\\\\* -Destination artifact"\n'
        )
        step = record.jobs[0].steps[0]
        self.assertEqual(
            step.run_value.value, "Copy-Item -Path .\\docs\\* -Destination artifact"
        )
        self.assertEqual(step.shell_family, "powershell")

    def test_shell_templates_and_dynamic_shells_are_explicit(self):
        record = self.parse_text(
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - {run: echo ready, shell: 'bash --noprofile -e {0}'}\n"
            "      - {run: echo ready, shell: '${{ matrix.shell }}'}\n"
            "      - {run: echo ready, shell: './runner {0}'}\n"
        )
        first, dynamic, custom = record.jobs[0].steps
        self.assertEqual(first.effective_shell, "bash --noprofile -e {0}")
        self.assertEqual(first.shell_family, "posix")
        self.assertEqual(dynamic.shell_family, "unknown")
        self.assertEqual(custom.shell_family, "unknown")

    def test_merge_and_flow_duplicate_fail_closed(self):
        for source, kind in (
            (
                "base: &base {path: docs/api/html}\n"
                "jobs: {j: {runs-on: ubuntu-latest, steps: "
                "[{uses: actions/upload-artifact@v4, with: {<<: *base}}]}}\n",
                "unsupported",
            ),
            (
                "jobs: {j: {runs-on: ubuntu-latest, steps: "
                "[{uses: actions/upload-artifact@v4, "
                "with: {path: build/, path: docs/api/html}}]}}\n",
                "parse",
            ),
        ):
            with self.subTest(source=source):
                with self.assertRaises(WorkflowReadError) as error:
                    self.parse_text(source)
                self.assertEqual(error.exception.kind, kind)

    def test_reusable_job_inputs_are_retained(self):
        record = self.parse_text(
            "jobs:\n"
            "  delegated:\n"
            "    uses: ./workflow.yml\n"
            "    with: {path: docs/api/html}\n"
        )
        job = record.jobs[0]
        self.assertEqual(job.uses.value, "./workflow.yml")
        self.assertEqual(dict(job.with_inputs)["path"].value, "docs/api/html")
        self.assertEqual(job.steps, ())

    def test_non_string_run_and_step_shape_fail(self):
        for source in (
            "jobs: {j: {steps: [{run: 42}]}}\n",
            "jobs: {j: {steps: [echo safe]}}\n",
            "jobs: {j: {steps: [{uses: a, run: echo safe}]}}\n",
            "jobs: {j: {steps: {run: echo safe}}}\n",
            "jobs: {}\n",
            "jobs: {j: {runs-on: ubuntu-latest, steps: []}}\n",
            "jobs: {j: {runs-on: ubuntu-latest, with: {x: y}, steps: [{run: echo ok}]}}\n",
        ):
            with self.subTest(source=source):
                with self.assertRaises(WorkflowReadError) as error:
                    self.parse_text(source)
                self.assertEqual(error.exception.kind, "parse")

    def test_extra_document_duplicate_and_cycle_fail_closed(self):
        for source in (
            "jobs: {j: {steps: [{run: echo safe}]}}\n---\nfoo: bar\n",
            "jobs: {j: {steps: [{run: echo safe}], steps: []}}\n",
            "jobs: &loop {j: *loop}\n",
        ):
            with self.subTest(source=source):
                with self.assertRaises(WorkflowReadError) as error:
                    self.parse_text(source)
                self.assertEqual(error.exception.kind, "parse")

    def test_duplicate_unrelated_key_and_explicit_tag_fail_closed(self):
        for source, kind in (
            ("name: first\nname: second\njobs: {}\n", "parse"),
            ("jobs: !!python/object/apply:os.system []\n", "unsupported"),
            ("jobs: !!str {j: {runs-on: ubuntu-latest, steps: [{run: echo ok}]}}\n", "unsupported"),
        ):
            with self.subTest(source=source):
                with self.assertRaises(WorkflowReadError) as error:
                    self.parse_text(source)
                self.assertEqual(error.exception.kind, kind)

    def test_no_workflow_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(WorkflowReadError) as error:
                read_workflows(Path(temp))
            self.assertEqual(error.exception.kind, "parse")

    def test_invalid_utf8_fails_as_parse_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "workflow.yml"
            path.write_bytes(b"jobs: \xff\n")
            with self.assertRaises(WorkflowReadError) as error:
                read_workflow(path)
            self.assertEqual(error.exception.kind, "parse")
            self.assertIn(str(path), str(error.exception))


if __name__ == "__main__":
    unittest.main()
