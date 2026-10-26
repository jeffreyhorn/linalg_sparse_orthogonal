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
        for path in (
            "docs/", "./DoCs/**", ".\\DoCs\\*", "**", ".",
            "../docs/api-reference.md", "../**", "build/../../**",
        ):
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

    def test_upload_actions_require_explicit_source_paths(self):
        for action in ("actions/upload-pages-artifact@v3", "actions/upload-artifact@v4"):
            with self.subTest(action=action):
                step = "      - run: cp -R docs _site/\n" + f"      - uses: {action}\n"
                self.assertIn("requires explicit source path", self.check_text(self.workflow(step)))
                explicit = (
                    f"      - uses: {action}\n"
                    "        with:\n          path: build/site\n"
                )
                self.assertIsNone(self.check_text(self.workflow(explicit)))

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

    def test_working_directory_resolves_staging_paths(self):
        cases = (
            "defaults:\n  run:\n    working-directory: docs\n"
            "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: cp -R . ../artifact/\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n",
            "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
            "    defaults:\n      run:\n        working-directory: docs\n"
            "    steps:\n      - run: cp -R . ../artifact/\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n",
            self.workflow(
                "      - run: cp -R . ../artifact/\n"
                "        working-directory: docs\n"
                "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
            ),
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertIn("stages docs", self.check_text(case))
        benign = self.workflow(
            "      - run: cp -R output ../artifact/\n"
            "        working-directory: build\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        )
        self.assertIsNone(self.check_text(benign))
        unresolved = self.workflow(
            "      - run: cp -R output artifact/\n"
            "        working-directory: $" "{{ inputs.directory }}\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        )
        self.assertIn("dynamic staging paths", self.check_text(unresolved))

    def test_posix_operators_keep_quoted_punctuation_distinct(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        rejecting = self.workflow(
            "      - run: |\n          echo ready;\n          cp -R docs artifact/\n" + upload
        )
        self.assertIn("stages docs", self.check_text(rejecting))
        benign = self.workflow("      - run: echo ';' cp -R docs artifact/\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_unquoted_substitution_in_staging_source_fails_closed(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        rejecting = self.workflow("      - run: cp -R $(printf docs) artifact/\n" + upload)
        self.assertIn("dynamic staging paths", self.check_text(rejecting))
        benign = self.workflow("      - run: echo $(printf docs)\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_source_glob_staging_fails_closed(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        for source in ("do*", "doc?", "./do[cs]*"):
            with self.subTest(source=source):
                step = f"      - run: cp -R {source} artifact/\n" + upload
                self.assertIn("dynamic staging paths", self.check_text(self.workflow(step)))
        for source in ("build/", "build/*"):
            with self.subTest(source=source):
                benign = self.workflow(f"      - run: cp -R {source} artifact/\n" + upload)
                self.assertIsNone(self.check_text(benign))

    def test_copy_target_directory_forms(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        for command in (
            "cp -R -t artifact docs", "cp -R --target-directory artifact docs",
            "cp -R --target-directory=artifact docs", "cp -R -tartifact docs",
        ):
            with self.subTest(command=command):
                self.assertIn("stages docs", self.check_text(self.workflow(f"      - run: {command}\n" + upload)))
        benign = self.workflow("      - run: cp -R -t artifact build\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_powershell_path_arrays_are_scanned_per_source(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact.zip\n"
        rejecting = self.workflow(
            "      - shell: pwsh\n"
            "        run: Compress-Archive -Path build,docs -DestinationPath artifact.zip\n"
            + upload, runner="windows-latest",
        )
        self.assertIn("archives docs", self.check_text(rejecting))
        benign = self.workflow(
            "      - shell: pwsh\n"
            "        run: Compress-Archive -Path 'build,docs' -DestinationPath artifact.zip\n"
            + upload, runner="windows-latest",
        )
        self.assertIsNone(self.check_text(benign))

    def test_powershell_mixed_named_and_positional_copy_arguments(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        for command in (
            "Copy-Item -Recurse docs -Destination artifact/",
            "Copy-Item -Recurse -Path docs artifact/",
        ):
            with self.subTest(command=command):
                step = f"      - shell: pwsh\n        run: {command}\n" + upload
                self.assertIn("stages docs", self.check_text(self.workflow(step, runner="windows-latest")))
        benign = self.workflow(
            "      - shell: pwsh\n        run: Copy-Item -Recurse -Path build artifact/\n" + upload,
            runner="windows-latest",
        )
        self.assertIsNone(self.check_text(benign))

    def test_all_shell_publisher_sources_are_checked(self):
        archive = "      - run: tar -czf artifact.tgz docs/\n"
        publishers = (
            "gh release upload v1 build/safe.txt artifact.tgz",
            "gsutil cp build/safe.txt artifact.tgz gs://example/release",
            "scp build/safe.txt artifact.tgz user@example:/release",
            "rsync build/safe.txt artifact.tgz user@example:/release",
        )
        for command in publishers:
            with self.subTest(command=command):
                self.assertIn(
                    "archives docs",
                    self.check_text(self.workflow(archive + f"      - run: {command}\n")),
                )
                safe = command.replace(" artifact.tgz", " build/other.txt")
                self.assertIsNone(self.check_text(self.workflow(f"      - run: {safe}\n")))

    def test_scp_switches_and_value_options_keep_asset_source(self):
        archive = "      - run: tar -czf artifact.tgz docs/\n"
        for options in ("-p", "-C", "-P 2222", "-c aes256-ctr", "-P 2222 -p"):
            with self.subTest(options=options):
                command = f"scp {options} artifact.tgz user@example:/release"
                self.assertIn("archives docs", self.check_text(self.workflow(archive + f"      - run: {command}\n")))
        benign = self.workflow("      - run: scp -p build/safe.txt user@example:/release\n")
        self.assertIsNone(self.check_text(benign))

    def test_nested_staging_destination_and_parent_copy(self):
        cases = (
            "      - run: cp -R docs artifact/site\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n",
            "      - run: cp -R docs artifact/site\n"
            "      - run: cp -R artifact/ bundle/\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: bundle/\n",
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertIn("stages docs", self.check_text(self.workflow(case)))
        benign = self.workflow(
            "      - run: cp -R docs artifact/site\n"
            "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact_other/\n"
        )
        self.assertIsNone(self.check_text(benign))

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
