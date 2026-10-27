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

    def test_effective_working_directory_generated_reference_is_rejected(self):
        cases = (
            self.workflow("      - run: ls .\n        working-directory: docs/api/html\n"),
            (
                "name: policy fixture\non: [push]\njobs:\n  build:\n"
                "    runs-on: ubuntu-latest\n    defaults:\n      run:\n"
                "        working-directory: docs/api/html\n    steps:\n      - run: ls .\n"
            ),
            (
                "name: policy fixture\non: [push]\ndefaults:\n  run:\n"
                "    working-directory: docs/api/html\njobs:\n  build:\n"
                "    runs-on: ubuntu-latest\n    steps:\n      - run: ls .\n"
            ),
        )
        for case in cases:
            with self.subTest(case=case):
                error = self.check_text(case)
                self.assertIsNotNone(error)
                self.assertIn("case.yml:", error)
                self.assertIn("jobs.build.steps[0]", error)
                self.assertIn("generated API output path in working-directory", error)
        self.assertIsNone(self.check_text(self.workflow("      - run: ls .\n        working-directory: build\n")))

    def test_broad_publication_paths_are_rejected(self):
        for path in (
            "docs/", "./DoCs/**", ".\\DoCs\\*", "**", ".",
            "../docs/api-reference.md", "../**", "build/../../**", "**/*.html",
        ):
            with self.subTest(path=path):
                step = (
                    "      - uses: actions/upload-artifact@v4\n"
                    f"        with:\n          path: '{path}'\n"
                )
                error = self.check_text(self.workflow(step, runner="windows-latest"))
                self.assertIsNotNone(error)
                self.assertIn("publishes docs or repository roots", error)

    def test_absolute_publication_paths_fail_closed(self):
        for path in (
            "/home/runner/work/linalg_sparse_orthogonal/linalg_sparse_orthogonal/**",
            "C:\\work\\linalg_sparse_orthogonal\\**",
        ):
            with self.subTest(path=path):
                upload = "      - uses: actions/upload-artifact@v4\n" + f"        with:\n          path: '{path}'\n"
                self.assertIn("publishes docs or repository roots", self.check_text(self.workflow(upload)))
        shell = self.workflow("      - run: aws s3 sync /home/runner/work/checkout s3://example/site\n")
        self.assertIn("publishes docs or repository roots", self.check_text(shell))
        benign = self.workflow("      - uses: actions/upload-artifact@v4\n        with:\n          path: build/output\n")
        self.assertIsNone(self.check_text(benign))

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

    def test_posix_assignment_prefixes_preserve_command_identity(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        for prefix in ("LC_ALL=C", "LC_ALL='C'"):
            with self.subTest(prefix=prefix):
                step = f"      - run: {prefix} cp -R docs artifact/\n" + upload
                self.assertIn("stages docs", self.check_text(self.workflow(step)))
        publisher = "      - run: LC_ALL=C aws s3 cp docs s3://example/site\n"
        self.assertIn("publishes docs or repository roots", self.check_text(self.workflow(publisher)))
        benign = self.workflow("      - run: LC_ALL=C cp -R build/ artifact/\n" + upload)
        self.assertIsNone(self.check_text(benign))
        quoted = self.workflow("      - run: |\n          \"LC_ALL=C\" cp -R docs artifact/\n" + upload)
        self.assertIsNone(self.check_text(quoted))

    def test_redirection_does_not_replace_copy_destination(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        for redirect in (">/dev/null", "> /dev/null", "2>/dev/null"):
            with self.subTest(redirect=redirect):
                step = f"      - run: cp -R docs artifact/ {redirect}\n" + upload
                self.assertIn("stages docs", self.check_text(self.workflow(step)))
        benign = self.workflow("      - run: echo 'cp -R docs artifact/ > /dev/null'\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_unquoted_substitution_in_staging_source_fails_closed(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        rejecting = self.workflow("      - run: cp -R $(printf docs) artifact/\n" + upload)
        self.assertIn("dynamic staging paths", self.check_text(rejecting))
        benign = self.workflow("      - run: echo $(printf docs)\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_backtick_substitution_in_staging_source_fails_closed(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        rejecting = self.workflow("      - run: cp -R `printf docs` artifact/\n" + upload)
        self.assertIn("dynamic staging paths", self.check_text(rejecting))
        benign = self.workflow("      - run: cp -R '`printf docs`' artifact/\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_tar_directory_context_applies_to_members_only(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact.tgz\n"
        for command in (
            "tar -C docs -czf artifact.tgz api",
            "tar -czf artifact.tgz -C docs api",
            "tar --directory=docs --file=artifact.tgz api",
        ):
            with self.subTest(command=command):
                self.assertIn("archives docs", self.check_text(self.workflow(f"      - run: {command}\n" + upload)))
        benign = self.workflow("      - run: tar -C build -czf artifact.tgz api\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_tar_ancestor_members_include_docs(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact.tar\n"
        for source in ("..", "../.."):
            with self.subTest(source=source):
                step = f"      - run: tar -cf artifact.tar {source}\n" + upload
                self.assertIn("archives docs", self.check_text(self.workflow(step)))
        benign = self.workflow("      - run: tar -cf artifact.tar build/\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_tar_member_names_are_not_legacy_option_bundles(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact.tar\n"
        for member in ("files", "Doxyfile"):
            with self.subTest(member=member):
                step = f"      - run: tar -cf artifact.tar docs/ {member}\n" + upload
                self.assertIn("archives docs", self.check_text(self.workflow(step)))
        benign = self.workflow("      - run: tar -cf artifact.tar build/ files\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_tar_stdout_archive_fails_closed_on_upload(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact.tgz\n"
        rejecting = self.workflow("      - run: tar -czf - docs/ > artifact.tgz\n" + upload)
        self.assertIn("dynamic staging paths", self.check_text(rejecting))
        benign = self.workflow("      - run: tar -czf - build/ > artifact.tgz\n" + upload)
        self.assertIsNone(self.check_text(benign))

    def test_zip_value_options_and_stdout_archive(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact.zip\n"
        for command, message in (
            ("zip -r -b build artifact.zip docs/", "archives docs"),
            ("zip -r - docs/ > artifact.zip", "dynamic staging paths"),
        ):
            with self.subTest(command=command):
                self.assertIn(message, self.check_text(self.workflow(f"      - run: {command}\n" + upload)))
                benign = command.replace("docs/", "build/")
                self.assertIsNone(self.check_text(self.workflow(f"      - run: {benign}\n" + upload)))

    def test_powershell_trailing_value_options(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        for command in (
            "Copy-Item -Recurse docs artifact -ErrorAction Stop",
            "Copy-Item -Path docs -Destination artifact -ErrorAction Stop",
        ):
            with self.subTest(command=command):
                step = f"      - shell: pwsh\n        run: {command}\n" + upload
                self.assertIn("stages docs", self.check_text(self.workflow(step, runner="windows-latest")))
                benign = step.replace(" docs ", " build ")
                self.assertIsNone(self.check_text(self.workflow(benign, runner="windows-latest")))
        unsupported = self.workflow(
            "      - shell: pwsh\n"
            "        run: Copy-Item -Recurse docs artifact -Unsupported value\n" + upload,
            runner="windows-latest",
        )
        self.assertIn("dynamic staging paths", self.check_text(unsupported))

    def test_cd_changes_staging_directory_within_run(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        rejecting = self.workflow("      - run: cd docs; cp -R . ../artifact/\n" + upload)
        self.assertIn("stages docs", self.check_text(rejecting))
        benign = self.workflow("      - run: cd build; cp -R output ../artifact/\n" + upload)
        self.assertIsNone(self.check_text(benign))
        unresolved = self.workflow("      - run: cd $WORKDIR; cp -R output ../artifact/\n" + upload)
        self.assertIn("dynamic staging paths", self.check_text(unresolved))
        publisher = self.workflow("      - run: cd docs; aws s3 sync . s3://example/site\n")
        self.assertIn("publishes docs or repository roots", self.check_text(publisher))

    def test_subshell_directory_does_not_leak_into_following_commands(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        rejecting = self.workflow("      - run: (cd build); cp -R docs artifact/\n" + upload)
        self.assertIn("stages docs", self.check_text(rejecting))
        benign = self.workflow("      - run: (cd docs); cp -R build artifact/\n" + upload)
        self.assertIsNone(self.check_text(benign))
        nested = self.workflow("      - run: (cd build; (cd docs)); cp -R docs artifact/\n" + upload)
        self.assertIn("stages docs", self.check_text(nested))

    def test_powershell_location_changes_affect_staging_paths(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        for change in ("Set-Location docs", "cd docs", "Set-Location -Path docs"):
            with self.subTest(change=change):
                step = (
                    "      - shell: pwsh\n"
                    f"        run: {change}; Copy-Item -Recurse * -Destination ../artifact/\n" + upload
                )
                self.assertIn("stages docs", self.check_text(self.workflow(step, runner="windows-latest")))
        benign = (
            "      - shell: pwsh\n"
            "        run: Set-Location build; Copy-Item -Recurse * -Destination ../artifact/\n" + upload
        )
        self.assertIsNone(self.check_text(self.workflow(benign, runner="windows-latest")))
        unresolved = (
            "      - shell: pwsh\n"
            "        run: Set-Location $env:DIR; Copy-Item -Recurse * -Destination ../artifact/\n" + upload
        )
        self.assertIn("dynamic staging paths", self.check_text(self.workflow(unresolved, runner="windows-latest")))

    def test_source_glob_staging_fails_closed(self):
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: artifact/\n"
        for source in ("do*", "doc?", "./do[cs]*", "*/*/*/*_8h.html"):
            with self.subTest(source=source):
                step = f"      - run: cp -R {source} artifact/\n" + upload
                self.assertIn("dynamic staging paths", self.check_text(self.workflow(step)))
        for source in ("build/", "build/*", "build/*/*/*_8h.html"):
            with self.subTest(source=source):
                benign = self.workflow(f"      - run: cp -R {source} artifact/\n" + upload)
                self.assertIsNone(self.check_text(benign))

    def test_absolute_staging_sources_fail_closed(self):
        checkout = "/home/runner/work/linalg_sparse_orthogonal/linalg_sparse_orthogonal"
        for command, upload_path in (
            (f"tar -cf artifact.tar {checkout}", "artifact.tar"),
            (f"cp -R {checkout} artifact/", "artifact/"),
        ):
            with self.subTest(command=command):
                upload = (
                    "      - uses: actions/upload-artifact@v4\n"
                    f"        with:\n          path: {upload_path}\n"
                )
                step = f"      - run: {command}\n" + upload
                self.assertIn("unbounded staging source", self.check_text(self.workflow(step)))
                benign = f"      - run: {command.replace(checkout, 'build/')}\n" + upload
                self.assertIsNone(self.check_text(self.workflow(benign)))

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

    def test_leading_publisher_options_keep_source_operands(self):
        stage = "      - run: cp -R docs artifact/\n"
        commands = (
            "aws s3 cp --recursive artifact/ s3://example/site",
            "aws s3 sync --delete artifact/ s3://example/site",
            "rclone copy --progress artifact/ remote:site",
            "wrangler pages deploy --project-name site artifact/",
            "vercel deploy --prod artifact/",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIn("stages docs", self.check_text(self.workflow(stage + f"      - run: {command}\n")))
                safe = command.replace("artifact/", "build/")
                self.assertIsNone(self.check_text(self.workflow(f"      - run: {safe}\n")))

    def test_home_publication_source_fails_closed_with_working_directory(self):
        for working_directory in ("", "        working-directory: build\n"):
            with self.subTest(working_directory=working_directory):
                step = (
                    "      - run: aws s3 sync ~ s3://example/release\n"
                    + working_directory
                )
                self.assertIn("publishes docs or repository roots", self.check_text(self.workflow(step)))
        safe = self.workflow("      - run: aws s3 sync build/ s3://example/release\n")
        self.assertIsNone(self.check_text(safe))

    def test_posix_positional_and_special_publication_parameters_are_dynamic(self):
        for source in ('"$1"', '"$@"', '"$*"'):
            with self.subTest(source=source):
                step = f"      - run: set -- docs; aws s3 sync {source} s3://example/site\n"
                self.assertIn("dynamic publication path", self.check_text(self.workflow(step)))
        action = self.workflow("      - uses: actions/upload-artifact@v4\n        with:\n          path: '$1'\n")
        self.assertIn("dynamic or unresolved publication path", self.check_text(action))
        benign = self.workflow("      - run: aws s3 sync build/ s3://example/site\n")
        self.assertIsNone(self.check_text(benign))

    def test_labeled_release_asset_preserves_local_source(self):
        archive = "      - run: tar -czf artifact.tgz docs/\n"
        rejecting = self.workflow(archive + "      - run: gh release upload v1 artifact.tgz#API\n")
        self.assertIn("archives docs", self.check_text(rejecting))
        benign = self.workflow("      - run: gh release upload v1 build/safe.tgz#API\n")
        self.assertIsNone(self.check_text(benign))

    def test_root_archive_matches_double_star_upload_glob(self):
        archive = "      - run: tar -czf artifact.tgz docs/\n"
        upload = "      - uses: actions/upload-artifact@v4\n        with:\n          path: '**/*.tgz'\n"
        self.assertIn("archives docs", self.check_text(self.workflow(archive + upload)))
        benign = self.workflow("      - run: tar -czf artifact.tgz build/\n" + upload)
        self.assertIsNone(self.check_text(benign))

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
