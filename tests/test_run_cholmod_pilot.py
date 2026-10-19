#!/usr/bin/env python3
"""Synthetic parser and failure-path checks; does not run the external solver."""

import importlib.util
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

    def test_schema_gate_blocks_before_build_or_solver(self) -> None:
        for scenario in ("missing_schema", "missing_validator", "failing_self_test"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                prefix = root / "suite-sparse" / "7.12.2"
                library = prefix / "lib/libcholmod.5.3.4.dylib"
                library.parent.mkdir(parents=True)
                library.touch()
                schema = root / "external_baseline_evidence_v1.json"
                validator = root / "validate_external_baseline_evidence.py"
                if scenario != "missing_schema":
                    schema.touch()
                if scenario != "missing_validator":
                    validator.touch()
                responses = {
                    "cholmod_version": "5.3.4",
                    "cholmod_prefix": str(prefix),
                    "cholmod_cflags": "",
                    "cholmod_libs": "",
                    "git_commit": "a" * 40,
                    "git_status": "",
                    "compiler_version": "test compiler",
                    "library_linkage": "test linkage",
                }

                def probe(_bundle, _attempts, label, _argv):
                    if label == "schema_preflight" and scenario == "failing_self_test":
                        raise pilot.PilotError("missing_dependency", "validator self-test exited 1")
                    return responses.get(label, "")

                def selected_hash(path):
                    return pilot.FIXTURE_SHA256 if path == pilot.FIXTURE else pilot.LIB_SHA256

                with mock.patch.object(pilot, "EVIDENCE_SCHEMA", schema), \
                     mock.patch.object(pilot, "EVIDENCE_VALIDATOR", validator), \
                     mock.patch.object(pilot, "sha256", side_effect=selected_hash), \
                     mock.patch.object(pilot, "checked_probe", side_effect=probe) as probes, \
                     mock.patch.object(pilot, "command") as solver_command:
                    record = pilot.run(root)
                labels = [call.args[2] for call in probes.call_args_list]
                self.assertNotIn("project_library_build", labels)
                self.assertNotIn("compile_helper", labels)
                solver_command.assert_not_called()
                self.assertEqual(record["provisional_status"], "blocked")
                self.assertEqual(record["sides"], {})
                self.assertEqual(record["timing"], "not_run")
                if scenario == "failing_self_test":
                    self.assertEqual(record["status_reason"], "schema_validation_failed")
                    self.assertEqual(record["schema_validation"],
                                     {"status": "failed", "reason": "validator_self_test_failed"})
                    self.assertIn("schema_preflight", labels)
                else:
                    self.assertEqual(record["status_reason"], "schema_validation_unavailable")
                    self.assertEqual(record["schema_validation"]["reason"], "v1_schema_or_validator_missing")
                    self.assertNotIn("schema_preflight", labels)

    def test_missing_optional_dependency_yields_blocked_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(pilot, "checked_probe", side_effect=pilot.PilotError("missing_dependency", "no cholmod")):
                record = pilot.run(Path(temp))
        self.assertEqual(record["provisional_status"], "blocked")
        self.assertEqual(record["status_reason"], "missing_dependency")
        self.assertEqual(record["sides"], {})
        self.assertEqual(record["timing"], "not_run")

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
        self.assertEqual(probe.call_count, 4)


if __name__ == "__main__":
    unittest.main()
