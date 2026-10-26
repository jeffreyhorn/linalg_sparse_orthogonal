#!/usr/bin/env python3
"""Direct-publication policy checks on source-located workflow records."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from api_docs_workflow_policy import PolicyViolation, check_direct_policy  # noqa: E402
from api_docs_workflow_policy_inputs import read_policy_inputs  # noqa: E402
from test_api_docs_local_only_guard import run_guard, write_fixture  # noqa: E402


class DirectPolicyTests(unittest.TestCase):
    def check_text(self, text: str) -> str | None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "case.yml").write_text(text, encoding="utf-8")
            try:
                check_direct_policy(read_policy_inputs(root))
            except PolicyViolation as exc:
                return str(exc)
        return None

    @staticmethod
    def workflow(step: str, *, runner: str = "ubuntu-latest", env: str = "") -> str:
        return (
            "name: policy fixture\non: [push]\n"
            f"{env}jobs:\n  build:\n    runs-on: {runner}\n    steps:\n{step}"
        )

    def test_maintained_workflows_are_safe(self):
        check_direct_policy(read_policy_inputs(REPO_ROOT))

    def test_direct_generated_references_include_location(self):
        cases = (
            ("      - run: ls docs/./api/html\n", "run command"),
            ("      - run: ls docs/a'p'i/html\n", "run command"),
            ("      - uses: actions/upload-artifact@v4\n        with:\n          path: './DoCs\\API\\HTML'\n", "path"),
            ("      - uses: actions/upload-artifact@v4\n        with:\n          name: docs/api/html\n          path: build/output\n", "name"),
        )
        for step, field in cases:
            with self.subTest(step=step):
                error = self.check_text(self.workflow(step))
                self.assertIsNotNone(error)
                self.assertIn("case.yml:", error)
                self.assertIn("jobs.build.steps[0]", error)
                self.assertIn("generated API output path", error)
                self.assertIn(field, error)

    def test_env_generated_reference_is_rejected(self):
        error = self.check_text(self.workflow("      - run: echo ready\n", env="env: {API_PATH: docs/api/html}\n"))
        self.assertIn("generated API output path in API_PATH", error)

    def test_broad_publication_paths_are_rejected(self):
        for path in ("docs/", "./DoCs/**", ".\\DoCs\\*", "**", ".", "../docs/api-reference.md"):
            with self.subTest(path=path):
                step = (
                    "      - uses: actions/upload-artifact@v4\n"
                    f"        with:\n          path: '{path}'\n"
                )
                error = self.check_text(self.workflow(step, runner="windows-latest"))
                self.assertIsNotNone(error)
                self.assertIn("publishes docs or repository roots", error)

    def test_multiline_path_checks_every_entry(self):
        step = (
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n          path: |\n            build/output\n            docs/**\n"
        )
        self.assertIn("publishes docs or repository roots", self.check_text(self.workflow(step)))

    def test_publication_action_families_and_unknown_actions(self):
        actions = (
            "actions/upload-pages-artifact@v3",
            "actions/deploy-pages@v4",
            "actions/upload-release-asset@v1",
            "softprops/action-gh-release@v2",
            "./.github/actions/deploy-docs",
            "docker://vendor/docs-publisher:latest",
            "vendor/check@v1",
        )
        for action in actions:
            with self.subTest(action=action):
                step = f"      - uses: {action}\n        with:\n          path: docs/\n"
                self.assertIn(
                    "publishes docs or repository roots", self.check_text(self.workflow(step))
                )

    def test_flow_quoted_alias_path_is_checked(self):
        step = "      - {uses: 'actions/upload-artifact@v4', with: {path: *target}}\n"
        error = self.check_text(
            self.workflow(step, env="env: {ARTIFACT: &target '.\\DoCs\\API\\HTML'}\n")
        )
        self.assertIn("generated API output path", error)

    def test_dynamic_path_and_action_fail_closed(self):
        cases = (
            ("      - uses: actions/upload-artifact@v4\n        with:\n          path: ${{ env.OUTPUT }}\n", "dynamic or unresolved publication path"),
            ("      - uses: ${{ inputs.publisher }}\n        with:\n          path: build/output\n", "dynamic or unresolved action identity"),
            ("      - uses: vendor/action@v1\n        with:\n          path: $OUTPUT\n", "dynamic or unresolved publication path"),
        )
        for step, message in cases:
            with self.subTest(step=step):
                self.assertIn(message, self.check_text(self.workflow(step)))

    def test_unsupported_action_and_path_forms_fail_closed(self):
        cases = (
            ("      - uses: not-an-action\n", "unsupported action form"),
            ("      - uses: ./\n", "unsupported local action form"),
            ("      - uses: actions/upload-artifact@v4\n        with:\n          path: 123\n", "unsupported publication path"),
        )
        for step, message in cases:
            with self.subTest(step=step):
                self.assertIn(message, self.check_text(self.workflow(step)))

    def test_unrelated_actions_and_safe_paths_are_allowed(self):
        cases = (
            "      - uses: actions/checkout@v4\n        with:\n          path: docs/\n",
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: build/output/**\n",
            "      - uses: vendor/check@v1\n        with:\n          path: build/output\n",
            "      - run: echo ready # docs/api/html\n",
        )
        for step in cases:
            with self.subTest(step=step):
                self.assertIsNone(self.check_text(self.workflow(step)))

    def test_guard_rejects_former_dynamic_action_bypass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            (root / ".github" / "workflows" / "api-docs.yml").write_text(
                self.workflow("      - uses: ${{ inputs.publisher }}\n        with:\n          path: build/output\n"),
                encoding="utf-8",
            )
            result = run_guard(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dynamic or unresolved action identity", result.stderr)
            self.assertIn("jobs.build.steps[0]", result.stderr)

    def test_staged_docs_reach_matching_upload(self):
        cases = (
            ("cp -R docs artifact/", "artifact/", "stages docs"),
            ("mv docs artifact/", "artifact/", "stages docs"),
            ("tar -czf artifact.tgz docs/", "artifact.tgz", "archives docs"),
            ("zip -r artifact.zip docs/", "artifact.zip", "archives docs"),
            ("7z a artifact.7z docs/", "artifact.7z", "archives docs"),
            ("cp -R build/ docs/ artifact/", "artifact/", "stages docs"),
            ("tar cf artifact.tgz build/ docs/", "artifact.tgz", "archives docs"),
            ("zip -r artifact.zip build/ docs/", "artifact.zip", "archives docs"),
            ("(cp -R docs artifact/)", "artifact/", "stages docs"),
        )
        for command, path, message in cases:
            with self.subTest(command=command):
                step = (
                    f"      - run: {command}\n"
                    "      - uses: actions/upload-artifact@v4\n"
                    f"        with:\n          path: {path}\n"
                )
                error = self.check_text(self.workflow(step))
                self.assertIn(message, error)
                self.assertIn("jobs.build.steps[0]", error)

    def test_powershell_staging_preserves_windows_paths(self):
        step = (
            "      - shell: pwsh\n"
            "        run: |\n"
            "          Copy-Item -Recurse -Path `\n"
            "            .\\docs\\* -Destination artifact\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n          path: artifact/\n"
        )
        self.assertIn("stages docs", self.check_text(self.workflow(step, runner="windows-latest")))

    def test_staging_chain_and_shell_publication(self):
        cases = (
            "      - run: cp -R docs staging/\n"
            "      - run: mv staging artifact/\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n          path: artifact/\n",
            "      - run: |\n"
            "          cp -R docs artifact/\n"
            "          aws s3 cp artifact/ s3://example/docs/ --recursive\n",
            "      - run: cp -R docs artifact/\n"
            "      - uses: peaceiris/actions-gh-pages@v4\n"
            "        with:\n          publish_dir: artifact/\n",
            "      - run: cp -R docs artifact/\n"
            "      - run: gh release upload v1 artifact.zip\n",
        )
        for step in cases[:3]:
            with self.subTest(step=step):
                self.assertIn("stages docs", self.check_text(self.workflow(step)))
        self.assertIsNone(self.check_text(self.workflow(cases[3])))
        matching_release = cases[3].replace("artifact.zip", "artifact/")
        self.assertIn("stages docs", self.check_text(self.workflow(matching_release)))

    def test_staging_keeps_command_and_job_boundaries(self):
        cases = (
            "      - run: cp -R docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: build/\n",
            "      - run: echo 'cp -R docs artifact/'\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n",
            "      - run: |\n          # cp -R docs artifact/\n          cp build artifact/\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n",
            "      - run: >\n          echo ready # explanation\n          cp -R docs artifact/\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n",
        )
        for step in cases:
            with self.subTest(step=step):
                self.assertIsNone(self.check_text(self.workflow(step)))
        separate_jobs = (
            "name: scoped\non: [push]\njobs:\n"
            "  one:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: cp -R docs artifact/\n"
            "  two:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n          path: artifact/\n"
        )
        self.assertIsNone(self.check_text(separate_jobs))

    def test_day9_canonical_and_command_edges(self):
        rejected = (
            ("      - uses: vendor/custom@v1\n        with:\n          path: '.\\out\\..\\docs\\**'\n", "publishes docs"),
            ("      - uses: vendor/custom@v1\n        with:\n          path: './build/${{ inputs.segment }}/out'\n", "dynamic or unresolved"),
            ("      - run: aws s3 cp ./docs/ s3://example/docs --recursive\n", "publishes docs"),
            ("      - run: gsutil -m rsync -r ./docs/* gs://example\n", "publishes docs"),
            ("      - run: gh release upload v1 ./docs/**\n", "publishes docs"),
            ("      - run: netlify deploy --dir=docs\n", "publishes docs"),
            ("      - run: aws s3 cp \"$DOCS_ROOT\" s3://example/docs\n", "dynamic publication"),
        )
        for step, detail in rejected:
            with self.subTest(step=step):
                self.assertIn(detail, self.check_text(self.workflow(step)))
        allowed = (
            "      - uses: vendor/custom@v1\n        with:\n          path: './build/../out/**'\n",
            "      - run: aws s3 cp build/output s3://example/build\n",
            "      - run: echo 'aws s3 cp docs/ s3://example/docs'\n",
        )
        for step in allowed:
            with self.subTest(step=step):
                self.assertIsNone(self.check_text(self.workflow(step)))

    def test_guard_allows_unrelated_upload_after_docs_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            (root / ".github" / "workflows" / "api-docs.yml").write_text(
                self.workflow(
                    "      - run: cp -R docs artifact/\n"
                    "      - uses: actions/upload-artifact@v4\n"
                    "        with:\n          path: build/\n"
                ),
                encoding="utf-8",
            )
            result = run_guard(root)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
