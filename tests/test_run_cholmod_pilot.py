#!/usr/bin/env python3
"""Synthetic parser and failure-path checks; does not run the external solver."""

import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cholmod_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_cholmod_pilot", SCRIPT)
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def output(side: str) -> str:
    common = [
        "loaded_cholmod_path=/usr/local/lib/libcholmod.dylib",
        f"side={side}",
        "load_status=0",
        "matrix_nnz=10",
    ]
    if side == "project":
        fields = ["factor_status=0", "used_csc_path=0", "solve_status=0"]
    else:
        fields = [
            "csc_nnz=7", "csc_stype=-1", "csc_sorted=1", "csc_packed=1", "csc_itype=0",
            "analyze_status=0", "factor_status=1", "cholmod_status=0", "factor_minor=4",
            "ordering=0", "is_super=0", "is_ll=1", "solve_status=1", "cholmod_solve_status=0",
        ]
    return "\n".join(common + fields + ["x=1,1,1,1", ""])


class PilotParserTests(unittest.TestCase):
    def test_both_synthetic_sides_pass_independent_numerical_rules(self) -> None:
        for side in ("project", "baseline"):
            with self.subTest(side=side):
                parsed = pilot.parse_side_output(output(side), side)
                self.assertTrue(pilot.check_correctness(parsed["solution"])["pass"])

    def test_solver_status_failure_is_not_a_pass(self) -> None:
        changed = output("baseline").replace("factor_minor=4", "factor_minor=3")
        with self.assertRaises(pilot.PilotError) as caught:
            pilot.parse_side_output(changed, "baseline")
        self.assertEqual(caught.exception.code, "solver_status_failure")

    def test_malformed_and_duplicate_output_are_rejected(self) -> None:
        for changed in (output("project") + "x=1,1,1,1\n", output("project") + "noise\n"):
            with self.subTest(changed=changed[-20:]):
                with self.assertRaises(pilot.PilotError) as caught:
                    pilot.parse_side_output(changed, "project")
                self.assertEqual(caught.exception.code, "malformed_output")

    def test_nonfinite_and_wrong_solution_are_rejected(self) -> None:
        for value in ("nan,1,1,1", "1,1,1"):
            with self.subTest(value=value):
                with self.assertRaises(pilot.PilotError):
                    pilot.parse_side_output(output("project").replace("1,1,1,1", value), "project")

    def test_mismatched_side_and_input_key_are_rejected(self) -> None:
        for changed in (output("project").replace("side=project", "side=baseline"),
                        output("project").replace("matrix_nnz=10", "matrix_nnz=9")):
            with self.subTest(changed=changed):
                with self.assertRaises(pilot.PilotError) as caught:
                    pilot.parse_side_output(changed, "project")
                self.assertEqual(caught.exception.code, "identity_or_load_failure")

    def test_numerically_bad_solution_fails_threshold(self) -> None:
        self.assertFalse(pilot.check_correctness((0.0, 0.0, 0.0, 0.0))["pass"])

    def test_finite_solution_with_overflowed_denominator_is_rejected(self) -> None:
        with self.assertRaises(pilot.PilotError) as caught:
            pilot.check_correctness((1.1e307, 1.0, 1.0, 1.0))
        self.assertEqual(caught.exception.code, "nonfinite_metric")

    def test_timeout_preserves_partial_streams_and_differs_from_launch_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            timeout = subprocess.TimeoutExpired(["probe"], 120, output=b"partial\xff", stderr=b"warning\n")
            with mock.patch.object(pilot.subprocess, "run", side_effect=timeout):
                timed_out = pilot.command(bundle, "timed_out", ["probe"])
            self.assertEqual(timed_out["returncode"], 124)
            self.assertEqual(timed_out["failure_kind"], "timeout")
            self.assertEqual(timed_out["stdout"], "partial\ufffd")
            self.assertEqual((bundle / "timed_out.stdout.txt").read_bytes(), b"partial\xff")
            self.assertEqual((bundle / "timed_out.stderr.txt").read_bytes(), b"warning\n")
            for stream in ("stdout", "stderr"):
                self.assertEqual(timed_out["files"][stream]["sha256"],
                                 pilot.sha256(bundle / timed_out["files"][stream]["path"]))
            with mock.patch.object(pilot.subprocess, "run", side_effect=OSError("cannot launch")):
                launch_error = pilot.command(bundle, "launch_error", ["probe"])
            self.assertEqual(launch_error["returncode"], 127)
            self.assertEqual(launch_error["failure_kind"], "launch_error")
            self.assertIn("cannot launch", (bundle / "launch_error.stderr.txt").read_text(encoding="utf-8"))

    def test_completed_command_preserves_raw_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            stdout = b"ready\r\nvalue=\xff\r\n"
            stderr = b"warning\r\nbyte=\xfe\r\n"
            completed = subprocess.CompletedProcess(["probe"], 0, stdout=stdout, stderr=stderr)
            with mock.patch.object(pilot.subprocess, "run", return_value=completed) as run_command:
                entry = pilot.command(bundle, "completed", ["probe"])
            self.assertNotIn("text", run_command.call_args.kwargs)
            self.assertEqual(entry["returncode"], 0)
            self.assertEqual(entry["stdout"], "ready\r\nvalue=\ufffd")
            self.assertEqual((bundle / "completed.stdout.txt").read_bytes(), stdout)
            self.assertEqual((bundle / "completed.stderr.txt").read_bytes(), stderr)
            for stream in ("stdout", "stderr"):
                self.assertEqual(entry["files"][stream]["sha256"],
                                 pilot.sha256(bundle / entry["files"][stream]["path"]))

    def test_schema_gate_blocks_before_build_or_solver(self) -> None:
        for scenario in ("missing_schema", "missing_validator", "unreadable_schema",
                         "unreadable_validator", "failing_self_test"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                schema = root / "external_baseline_evidence_v1.json"
                validator = root / "validate_external_baseline_evidence.py"
                if scenario != "missing_schema":
                    schema.touch()
                if scenario != "missing_validator":
                    validator.touch()
                commands = []

                def completed_command(argv, **_kwargs):
                    commands.append(argv)
                    self.assertIn("--self-test", argv)
                    return subprocess.CompletedProcess(argv, 1, stdout=b"failed\r\n\xff",
                                                       stderr=b"negative fixture failed\r\n\xfe")

                original_open = Path.open
                def selective_open(path, *args, **kwargs):
                    if (scenario == "unreadable_schema" and path == schema) or \
                       (scenario == "unreadable_validator" and path == validator):
                        raise PermissionError("synthetic unreadable input")
                    return original_open(path, *args, **kwargs)

                with mock.patch.object(pilot, "EVIDENCE_SCHEMA", schema), \
                     mock.patch.object(pilot, "EVIDENCE_VALIDATOR", validator), \
                     mock.patch.object(Path, "open", selective_open), \
                     mock.patch.object(pilot.subprocess, "run", side_effect=completed_command):
                    record = pilot.run(root)
                self.assertEqual(len(commands), 1 if scenario == "failing_self_test" else 0)
                self.assertNotIn("project_library_build", record["commands"])
                self.assertNotIn("compile_helper", record["commands"])
                self.assertNotIn("project", record["commands"])
                self.assertNotIn("baseline", record["commands"])
                self.assertEqual(record["provisional_status"], "blocked")
                self.assertEqual(record["sides"], {})
                self.assertEqual(record["timing"], "not_run")
                if scenario == "failing_self_test":
                    self.assertEqual(record["status_reason"], "schema_validation_failed")
                    self.assertEqual(record["schema_validation"],
                                     {"status": "failed", "reason": "validator_self_test_failed"})
                    self.assertEqual(record["commands"]["schema_preflight"]["returncode"], 1)
                    preflight = record["commands"]["schema_preflight"]
                    for stream, expected in (("stdout", b"failed\r\n\xff"),
                                             ("stderr", b"negative fixture failed\r\n\xfe")):
                        path = root / preflight["files"][stream]["path"]
                        self.assertEqual(path.read_bytes(), expected)
                        self.assertEqual(preflight["files"][stream]["sha256"], pilot.sha256(path))
                else:
                    self.assertEqual(record["status_reason"], "schema_validation_unavailable")
                    reason = ("v1_schema_or_validator_unreadable" if scenario.startswith("unreadable") else
                              "v1_schema_or_validator_missing")
                    self.assertEqual(record["schema_validation"], {"status": "failed", "reason": reason})
                    self.assertNotIn("schema_preflight", record["commands"])

    def test_invalid_schema_preflight_stops_before_dependency_discovery(self) -> None:
        schema_data = json.loads(pilot.EVIDENCE_SCHEMA.read_text(encoding="utf-8"))
        relaxed = json.loads(json.dumps(schema_data))
        relaxed["additionalProperties"] = True
        for label, content, expected in (("malformed", b"{", "invalid_json"),
                                         ("relaxed", json.dumps(relaxed).encode("utf-8"),
                                          "implementation_pending /self_test/unknown_field")):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                schema = root / "schema.json"
                schema.write_bytes(content)
                with mock.patch.object(pilot, "EVIDENCE_SCHEMA", schema):
                    record = pilot.run(root)
                self.assertEqual(record["provisional_status"], "blocked")
                self.assertEqual(record["status_reason"], "schema_validation_failed")
                self.assertEqual(record["schema_validation"],
                                 {"status": "failed", "reason": "validator_self_test_failed"})
                self.assertEqual(set(record["commands"]), {"schema_preflight"})
                self.assertEqual(record["sides"], {})
                self.assertEqual(record["timing"], "not_run")
                entry = record["commands"]["schema_preflight"]
                self.assertNotEqual(entry["returncode"], 0)
                for stream in ("stdout", "stderr"):
                    path = root / entry["files"][stream]["path"]
                    self.assertEqual(entry["files"][stream]["sha256"], pilot.sha256(path))
                self.assertIn(expected, (root / entry["files"]["stderr"]["path"]).read_text(encoding="utf-8"))

    def test_passing_real_self_test_reaches_only_mocked_downstream_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fixture = root / "tests/data/symmetric_4.mtx"
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes(pilot.FIXTURE.read_bytes())
            prefix = root / "suite-sparse/7.12.2"
            library = prefix / "lib/libcholmod.5.3.4.dylib"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"synthetic CHOLMOD library\n")
            archive = root / "build/libsparse_lu_ortho.a"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"synthetic project archive\n")
            bundle = root / "attempt"
            bundle.mkdir()
            helper = bundle / "cholmod_pilot"
            seen = []
            real_run = subprocess.run

            def completed_command(argv, **kwargs):
                seen.append(argv)
                if "--self-test" in argv:
                    return real_run(argv, **kwargs)
                if argv[:2] == ["pkg-config", "--modversion"]:
                    stdout = b"5.3.4\n"
                elif argv[:2] == ["pkg-config", "--variable=prefix"]:
                    stdout = str(prefix).encode("utf-8") + b"\n"
                elif argv[:2] in (["pkg-config", "--cflags"], ["pkg-config", "--libs"]):
                    stdout = b""
                elif argv[:2] == ["git", "rev-parse"]:
                    stdout = b"a" * 40 + b"\n"
                elif argv[:2] == ["git", "status"]:
                    stdout = b""
                elif argv[:2] == ["cc", "--version"]:
                    stdout = b"synthetic compiler\n"
                elif argv[0] == "otool":
                    stdout = b"synthetic linkage\n"
                elif argv[0] == "make":
                    stdout = b"synthetic build\n"
                elif argv[0] == "cc" and "-o" in argv:
                    helper.write_bytes(b"synthetic helper\n")
                    stdout = b"synthetic compile\n"
                elif argv[0] == str(helper) and argv[1] in ("project", "baseline"):
                    stdout = output(argv[1]).replace("/usr/local/lib/libcholmod.dylib",
                                                      str(library)).encode("utf-8")
                else:
                    self.fail(f"unexpected pilot command: {argv}")
                return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

            with mock.patch.object(pilot, "ROOT", root), \
                 mock.patch.object(pilot, "FIXTURE", fixture), \
                 mock.patch.object(pilot, "FIXTURE_SHA256", pilot.sha256(fixture)), \
                 mock.patch.object(pilot, "LIB_SHA256", pilot.sha256(library)), \
                 mock.patch.object(pilot.subprocess, "run", side_effect=completed_command):
                record = pilot.run(bundle)
            self.assertEqual(record["schema_validation"],
                             {"status": "preflight_pass", "reason": "negative_fixtures_checked"})
            self.assertEqual(record["provisional_status"], "two_sided_correctness_candidate")
            self.assertEqual(record["timing"], "not_run")
            self.assertEqual(record["publication_review"], "pending")
            self.assertEqual(set(record["sides"]), {"project", "baseline"})
            self.assertTrue(all(side["metrics"]["pass"] for side in record["sides"].values()))
            self.assertEqual(list(record["commands"]), [
                "schema_preflight", "cholmod_version", "cholmod_prefix", "cholmod_cflags",
                "cholmod_libs", "git_commit", "git_status", "compiler_version",
                "library_linkage", "project_library_build", "compile_helper", "project", "baseline",
            ])
            self.assertIn("--self-test", seen[0])
            self.assertEqual([argv[1] for argv in seen[-2:]], ["project", "baseline"])

    def test_missing_optional_dependency_yields_blocked_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            labels = []
            def probe(_bundle, _attempts, label, _argv):
                labels.append(label)
                if label == "schema_preflight":
                    return "self-test PASS"
                raise pilot.PilotError("missing_dependency", "no cholmod")
            with mock.patch.object(pilot, "checked_probe", side_effect=probe):
                record = pilot.run(Path(temp))
        self.assertEqual(record["provisional_status"], "blocked")
        self.assertEqual(record["status_reason"], "missing_dependency")
        self.assertEqual(record["sides"], {})
        self.assertEqual(record["timing"], "not_run")
        self.assertEqual(record["schema_validation"],
                         {"status": "preflight_pass", "reason": "negative_fixtures_checked"})
        self.assertEqual(labels, ["schema_preflight", "cholmod_version"])

    def test_existing_attempt_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            output_dir = root / "build/external-baseline" / pilot.SELECTION_ID / "attempt-1"
            output_dir.mkdir(parents=True)
            marker = output_dir / "attempt.json"
            marker.write_bytes(b"prior attempt\n")
            argv = ["pilot", "--fixture", str(pilot.FIXTURE), "--output", str(output_dir),
                    "--correctness-only", "--execute"]
            with mock.patch.object(pilot, "ROOT", root), \
                 mock.patch.object(pilot.sys, "argv", argv), \
                 mock.patch.object(pilot.sys, "stderr", io.StringIO()) as error_stream, \
                 mock.patch.object(pilot, "run") as run_attempt:
                with self.assertRaises(SystemExit) as caught:
                    pilot.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("attempt output already exists", error_stream.getvalue())
            run_attempt.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"prior attempt\n")

    def test_changed_fixture_does_not_reach_dependency_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(pilot, "sha256", return_value="bad"):
                with mock.patch.object(pilot, "checked_probe") as probe:
                    record = pilot.run(Path(temp))
                    probe.assert_not_called()
        self.assertEqual(record["provisional_status"], "blocked")
        self.assertEqual(record["status_reason"], "fixture_hash_mismatch")

    def test_changed_baseline_version_blocks_before_solver(self) -> None:
        responses = {
            "schema_preflight": "self-test PASS",
            "cholmod_version": "5.3.3",
            "cholmod_prefix": "/usr/local/Cellar/suite-sparse/7.12.2",
            "cholmod_cflags": "",
            "cholmod_libs": "",
        }
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(pilot, "checked_probe", side_effect=lambda _, __, label, ___: responses[label]) as probe:
                record = pilot.run(Path(temp))
        self.assertEqual(record["provisional_status"], "blocked")
        self.assertEqual(record["status_reason"], "baseline_version_mismatch")
        self.assertEqual(record["sides"], {})
        self.assertEqual(record["timing"], "not_run")
        self.assertEqual(probe.call_count, 5)


if __name__ == "__main__":
    unittest.main()
