#!/usr/bin/env python3
"""V1 identity and strict structural checks; no external solve."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_external_baseline_evidence.py"
SCHEMA = ROOT / "tests/corpus/schemas/external_baseline_evidence_v1.json"
BLUEPRINT = ROOT / "docs/planning/EPIC_19/SPRINT_215/artifacts/day7-external-evidence-blueprint.example.json"
PLANNED = ROOT / "tests/corpus/fixtures/external_baseline_planned_v1.json"
CANONICAL = ROOT / "tests/corpus/fixtures/external_baseline_identity_v1.canonical.json"
SPEC = importlib.util.spec_from_file_location("validate_external_baseline_evidence", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def complete_identity() -> dict:
    return {
        "matrix_sha256": "ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335",
        "problem": {
            "class": "spd_linear_solve", "operation": "A*x=b", "rhs": [13, 14, 17, 16],
            "precision": "float64", "project_index_abi": "int32", "baseline_index_abi": "int32",
        },
        "project": {
            "git_commit": "a" * 40, "worktree_state": "clean",
            "patch_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "binary_sha256": "b" * 64, "compiler": "clang 1", "compile_flags": "-O2",
            "build_mode": "Release", "solver_api": ["sparse_cholesky_factor_opts", "sparse_cholesky_solve"],
            "backend": "linked_list", "ordering": "none",
        },
        "baseline": {
            "source_commit": "42151688813c45846a597edcb601435a0e38f3dd",
            "loaded_library_sha256": "c" * 64,
            "dependencies": [
                {"role": "openmp", "loaded_path": "/opt/libomp.dylib", "sha256": "d" * 64},
                {"role": "blas", "loaded_path": "/opt/blas.dylib", "sha256": "e" * 64},
            ],
            "build_options": "CHOLMOD_MODULES=all", "module_composition": ["cholmod", "amd"],
            "solver_api": ["cholmod_analyze", "cholmod_factorize", "cholmod_solve(CHOLMOD_A)"],
            "backend": "simplicial", "ordering": "natural", "postorder": False,
        },
        "threads": {"requested": {"blas": 1, "openmp": 1},
                    "observed": {"blas": 1, "openmp": 1}},
        "host": {"os": "Darwin", "os_build": "synthetic", "architecture": "x86_64",
                 "cpu": "test-\u00b5CPU"},
    }


def synthetic_observed() -> dict:
    record = validator.read_json(BLUEPRINT)
    record.update(record_id="synthetic-observed-v1", record_kind="run",
                  evidence_origin="generated_local", evidence_status="correctness_observed",
                  status_reason="two_sides_passed", run_id="synthetic-run-001",
                  cell_id_reason=None, claim_scope="one_cell_local_correctness_only")
    record["non_claims"] = record["non_claims"].replace("no_external_solver_result;", "")
    cell = record["cell"]
    cell["matrix"]["source_commit"] = "aa1c04eab3536031b9a92ee92587260520ce1913"
    cell["problem"].update(baseline_index_abi="int32", index_abi_reason=None)
    cell["project"].update(git_commit="a" * 40, worktree_state="clean",
                           patch_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                           compiler="synthetic clang", compile_flags="-O2", build_mode="Release",
                           binary_sha256="b" * 64, runtime_reason=None)
    cell["baseline"].update(loaded_library_path="/opt/libcholmod.dylib",
                            loaded_library_sha256="c" * 64, build_options="synthetic options",
                            module_composition=["cholmod", "amd"],
                            dependencies=[{"role": "blas", "loaded_path": "/opt/blas.dylib",
                                           "sha256": "d" * 64}], runtime_reason=None)
    for name in ("identity", "correctness", "measurement"):
        record["methodology"][name]["sha256"] = "e" * 64
    record["methodology"]["hash_reason"] = None
    record["environment"] = {
        "os": "Darwin", "os_build": "synthetic", "architecture": "x86_64", "cpu": "synthetic cpu",
        "compiler_runtime": "synthetic", "blas_identity": "synthetic blas",
        "openmp_identity": "synthetic openmp", "threads_requested": {"blas": 1, "openmp": 1},
        "threads_observed": {"blas": 1, "openmp": 1}, "command": ["synthetic", "test"],
        "cwd": "/tmp", "timestamp_utc": "2026-10-28T00:00:00Z",
        "host_fingerprint": "f" * 64, "clock_identity": None, "clock_resolution_ns": 1,
    }
    record["environment_reason"] = None
    metrics = validator._solution_metrics([1.0, 1.0, 1.0, 1.0])
    for name in ("project", "baseline"):
        side = record["correctness"][name]
        side.update(state="pass", phase_statuses={"load_status": 0, "factor_status": 0,
                                                  "solve_status": 0} if name == "project" else
                    {"load_status": 0, "analyze_status": 0, "factor_status": 1,
                     "solve_status": 1, "cholmod_status": 0},
                    factor_diagnostics={"used_csc_path": 0} if name == "project" else
                    {"factor_minor": 4, "ordering": 0, "is_super": 0, "csc_itype": 0},
                    solution=[1.0, 1.0, 1.0, 1.0], raw_artifact=f"{name}.json",
                    availability_reason=None, **metrics)
        record["artifacts"].append({"path": f"{name}.json", "role": f"{name}_raw",
                                    "media_type": "application/json", "sha256": "a" * 64})
    record["correctness"]["cross_solver_difference_inf"] = 0.0
    record["correctness"]["cross_solver_consistency_limit"] = 2 * metrics["oracle_limit"]
    record["timing"]["availability_reason"] = "correctness_only"
    record["cell_id"] = validator.canonical_cell_id(validator.identity_from_record(record))[1]
    return record


def synthetic_measured() -> tuple[dict, dict]:
    record = synthetic_observed()
    record["evidence_status"] = "measured"
    record["status_reason"] = "synthetic_timing_fixture"
    record["claim_scope"] = "one_cell_local_descriptive_latency_only"
    record["non_claims"] = record["non_claims"].replace("no_timing_result;", "")
    record["methodology"]["timing_policy"] = "eligible"
    record["environment"]["clock_identity"] = record["methodology"]["planned_clock"]
    def pairs(count: int, offset: int) -> list[dict]:
        return [{"order": "AB" if i % 2 == 0 else "BA", "project_status": "pass",
                 "baseline_status": "pass", "project_ns": offset + 1000 + i,
                 "baseline_ns": offset + 1200 + i} for i in range(count)]
    raw = {}
    windows = []
    for index, phase in enumerate(("factor_plus_first_solve", "factor_ready", "one_solve")):
        raw[phase] = {"warmup": pairs(5, index * 100), "timed": pairs(50, index * 100)}
        window = {"phase": phase, "warmup_pairs_observed": 5, "timed_pairs_observed": 50,
                  "raw_samples_artifact": "samples.json", "quartile_convention": "median_of_halves"}
        for side in ("project", "baseline"):
            window[side] = validator._summary([pair[f"{side}_ns"] for pair in raw[phase]["timed"]])
        windows.append(window)
    record["timing"].update(state="measured", warmup_pairs_observed=5, timed_pairs_observed=50,
                            raw_samples_artifact="samples.json", availability_reason=None,
                            median_ns={side: windows[0][side]["median_ns"] for side in ("project", "baseline")},
                            windows=windows)
    record["artifacts"].append({"path": "samples.json", "role": "timing_raw",
                                "media_type": "application/json", "sha256": "a" * 64})
    return record, raw


def synthetic_correctness_failed() -> dict:
    record = synthetic_observed()
    record["record_id"] = "synthetic-correctness-failed-v1"
    record["evidence_status"] = "correctness_failed"
    record["status_reason"] = "synthetic_oracle_gate_failed"
    side = record["correctness"]["baseline"]
    side.update(state="fail", solution=[0.0] * 4,
                availability_reason="synthetic_oracle_gate_failed")
    side.update(validator._solution_metrics(side["solution"]))
    record["correctness"].update(cross_solver_difference_inf=None,
                                 cross_solver_consistency_limit=None)
    record["claim_scope"] = "no_result"
    record["non_claims"] = "no_external_solver_result;" + record["non_claims"]
    return record


def synthetic_timing_invalid() -> dict:
    record = synthetic_observed()
    record["record_id"] = "synthetic-timing-invalid-v1"
    record["evidence_status"] = "timing_invalid"
    record["status_reason"] = "synthetic_resolution_limited"
    record["timing"].update(state="invalid", raw_samples_artifact="attempt.json",
                            availability_reason="synthetic_resolution_limited")
    record["artifacts"].append({"path": "attempt.json", "role": "timing_attempt",
                                "media_type": "application/json", "sha256": "a" * 64})
    return record


def stage_synthetic_bundle(record: dict, bundle: Path, raw_samples: dict | None = None) -> None:
    """Write synthetic fixture bytes only under a temporary bundle."""
    for name in ("identity", "correctness", "measurement"):
        method = record["methodology"][name]
        method["sha256"] = hashlib.sha256((ROOT / method["path"]).read_bytes()).hexdigest()
    for artifact in record["artifacts"]:
        path = bundle / artifact["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact["path"] == record["timing"]["raw_samples_artifact"] and raw_samples is not None:
            content = json.dumps(raw_samples, sort_keys=True).encode("utf-8")
        elif artifact["role"] in ("project_raw", "baseline_raw"):
            side = artifact["role"].removesuffix("_raw")
            content = json.dumps({key: record["correctness"][side][key]
                                  for key in validator.RAW_SIDE_KEYS}, sort_keys=True).encode("utf-8")
        else:
            content = b"synthetic-attempt-only\n"
        path.write_bytes(content)
        artifact["sha256"] = hashlib.sha256(content).hexdigest()


def synthetic_not_run() -> dict:
    record = validator.read_json(PLANNED)
    record.update(record_id="synthetic-s221-not-run", record_kind="run",
                  evidence_origin="generated_local", evidence_status="not_run",
                  status_reason="preflight_not_started", run_id="synthetic-attempt-not-run",
                  claim_scope="no_result")
    return record


def synthetic_execution_blocked() -> dict:
    record = synthetic_not_run()
    record.update(record_id="synthetic-s221-execution-blocked", evidence_status="blocked",
                  status_reason="cholmod_runtime_unavailable", block_gate="execution",
                  run_id="synthetic-attempt-blocked")
    record["correctness"]["baseline"].update(state="blocked",
                                               availability_reason="cholmod_runtime_unavailable")
    return record


def synthetic_non_comparable() -> dict:
    record = synthetic_not_run()
    record["record_id"] = "synthetic-non-comparable-v1"
    record["evidence_status"] = "non_comparable"
    record["status_reason"] = "synthetic_baseline_abi_unverified"
    record["mismatch_field"] = "baseline_index_abi"
    return record


class IdentityTests(unittest.TestCase):
    def test_canonical_hash_vector_and_dependency_sort(self) -> None:
        identity = complete_identity()
        payload, cell_id = validator.canonical_cell_id(identity)
        self.assertEqual(payload, CANONICAL.read_bytes().removesuffix(b"\n"))
        self.assertEqual(cell_id, "sha256:5ff34c9c3d5661942d98b94a4c5dabf62b10920bec03f364f69201126d54481d")
        self.assertIn(b"test-\\u00b5CPU", payload)
        self.assertLess(payload.index(b"/opt/blas.dylib"), payload.index(b"/opt/libomp.dylib"))
        reversed_identity = copy.deepcopy(identity)
        reversed_identity["baseline"]["dependencies"].reverse()
        self.assertEqual(validator.canonical_cell_id(reversed_identity), (payload, cell_id))

    def test_identity_change_changes_id(self) -> None:
        identity = complete_identity()
        original = validator.canonical_cell_id(identity)[1]
        identity["host"]["cpu"] = "different"
        self.assertNotEqual(validator.canonical_cell_id(identity)[1], original)

    def test_partial_or_ambiguous_identity_is_rejected(self) -> None:
        for mutator in (
            lambda item: item["project"].__setitem__("binary_sha256", None),
            lambda item: item["threads"]["observed"].__setitem__("blas", True),
            lambda item: item["baseline"]["dependencies"].append(item["baseline"]["dependencies"][0]),
            lambda item: item["baseline"]["dependencies"][1].__setitem__("role", "openmp"),
            lambda item: item["baseline"]["dependencies"][1].__setitem__("loaded_path", "/opt/libomp.dylib"),
            lambda item: item["baseline"]["dependencies"][0].__setitem__("loaded_path", "relative/lib.dylib"),
            lambda item: item["project"].__setitem__("git_commit", "a" * 8),
        ):
            with self.subTest(mutator=mutator):
                identity = complete_identity()
                mutator(identity)
                with self.assertRaises(validator.EvidenceError):
                    validator.canonical_cell_id(identity)


class InterfaceTests(unittest.TestCase):
    def test_schema_authority_and_blueprint_shape(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        blueprint = json.loads(BLUEPRINT.read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], blueprint["schema_version"])
        self.assertEqual(set(blueprint), set(schema["required"]))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["$defs"]["matrix"]["properties"]["source_commit"]["enum"],
                         ["aa1c04ea", "aa1c04eab3536031b9a92ee92587260520ce1913"])
        self.assertEqual(blueprint["record_kind"], "blueprint")
        self.assertEqual(blueprint["evidence_status"], "planned")
        self.assertIsNone(blueprint["cell_id"])

    def test_cli_self_test_and_planning_input_validate_without_solver_claim(self) -> None:
        for args in (("--self-test",), ("--input", str(BLUEPRINT))):
            with self.subTest(args=args):
                run = subprocess.run([sys.executable, str(SCRIPT), "--schema", str(SCHEMA), *args],
                                     capture_output=True, text=True, check=False)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("external-baseline-evidence-v1:", run.stdout)
                self.assertNotIn("correctness_observed", run.stdout)

    def test_negative_self_test_fails_if_schema_accepts_unknown_fields(self) -> None:
        schema = validator.read_json(SCHEMA)
        schema["additionalProperties"] = True
        with self.assertRaises(validator.EvidenceError) as caught:
            validator.run_self_test(schema, ROOT)
        self.assertEqual((caught.exception.code, caught.exception.pointer),
                         ("implementation_pending", "/self_test/unknown_field"))
        with tempfile.TemporaryDirectory() as temp:
            relaxed = Path(temp) / "relaxed-schema.json"
            relaxed.write_text(json.dumps(schema), encoding="utf-8")
            run = subprocess.run([sys.executable, str(SCRIPT), "--schema", str(relaxed),
                                  "--input", str(BLUEPRINT)], capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("implementation_pending /self_test/unknown_field", run.stderr)

    def test_duplicate_key_is_rejected_even_before_semantic_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            record = Path(temp) / "duplicate.json"
            record.write_text('{"record_id":"one","record_id":"two"}', encoding="utf-8")
            run = subprocess.run([sys.executable, str(SCRIPT), "--schema", str(SCHEMA),
                                  "--input", str(record)], capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 1)
            self.assertIn("duplicate_or_unknown_field", run.stderr)


class StructuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = validator.read_json(SCHEMA)
        cls.blueprint = validator.read_json(BLUEPRINT)

    def test_planning_blueprint_is_structurally_valid(self) -> None:
        validator.validate_shape(self.blueprint, self.schema)
        full_origin = copy.deepcopy(self.blueprint)
        full_origin["cell"]["matrix"]["source_commit"] = "aa1c04eab3536031b9a92ee92587260520ce1913"
        validator.validate_shape(full_origin, self.schema)

    def test_missing_unknown_mistyped_and_changed_selection_fields(self) -> None:
        cases = (
            (lambda r: r.pop("record_id"), "missing_required_field", "/record_id"),
            (lambda r: r["cell"]["matrix"].pop("rows"), "missing_required_field", "/cell/matrix/rows"),
            (lambda r: r.__setitem__("extra", 1), "duplicate_or_unknown_field", "/extra"),
            (lambda r: r["cell"]["matrix"].__setitem__("extra", 1),
             "duplicate_or_unknown_field", "/cell/matrix/extra"),
            (lambda r: r["cell"]["matrix"].__setitem__("rows", True), "exact_type", "/cell/matrix/rows"),
            (lambda r: r["cell"]["matrix"].__setitem__("rows", 4.0), "exact_type", "/cell/matrix/rows"),
            (lambda r: r.__setitem__("record_id", ""), "exact_type", "/record_id"),
            (lambda r: r.__setitem__("schema_version", "v2"), "unsupported_enum", "/schema_version"),
            (lambda r: r.__setitem__("evidence_status", "passing"), "unsupported_enum", "/evidence_status"),
            (lambda r: r["correctness"]["project"].__setitem__("state", "passing"),
             "unsupported_enum", "/correctness/project/state"),
            (lambda r: r["cell"]["matrix"].__setitem__("sha256", "f" * 64),
             "selection_mismatch", "/cell/matrix/sha256"),
            (lambda r: r["cell"]["problem"].__setitem__("operation", "A*x=y"),
             "selection_mismatch", "/cell/problem/operation"),
            (lambda r: r["cell"]["rhs"].__setitem__("values", [13, 14, 17, 15]),
             "selection_mismatch", "/cell/rhs/values"),
            (lambda r: r["cell"]["project"].__setitem__("backend", "csc"),
             "selection_mismatch", "/cell/project/backend"),
            (lambda r: r["cell"]["baseline"].__setitem__("candidate_binary_sha256", "ABC"),
             "missing_provenance", "/cell/baseline/candidate_binary_sha256"),
            (lambda r: r["cell"]["project"].__setitem__("git_commit", "deadbeef"),
             "missing_provenance", "/cell/project/git_commit"),
            (lambda r: r["artifacts"].append({"path": "../secret", "role": "log", "media_type": "text/plain",
                                                  "sha256": "a" * 64}),
             "exact_type", "/artifacts/0/path"),
            (lambda r: r["timing"].__setitem__("unit", "ms"),
             "unit_or_sample_contract", "/timing/unit"),
            (lambda r: r["timing"].__setitem__("warmup_pairs_observed", False),
             "exact_type", "/timing/warmup_pairs_observed"),
        )
        for mutate, code, pointer in cases:
            with self.subTest(code=code, pointer=pointer):
                record = copy.deepcopy(self.blueprint)
                mutate(record)
                with self.assertRaises(validator.EvidenceError) as caught:
                    validator.validate_shape(record, self.schema)
                self.assertEqual((caught.exception.code, caught.exception.pointer), (code, pointer))

    def test_nested_duplicate_nonfinite_and_invalid_utf8(self) -> None:
        cases = (
            (b'{"outer":{"a":1,"a":2}}', "duplicate_or_unknown_field", "/outer/a"),
            (b'{"outer":[{"metric":NaN}]}', "nonfinite_or_missing_value_abuse", "/outer/0/metric"),
            (b'{"outer":[{"metric":Infinity}]}', "nonfinite_or_missing_value_abuse", "/outer/0/metric"),
            (b'{"outer":[{"metric":1e999}]}', "nonfinite_or_missing_value_abuse", "/outer/0/metric"),
            (b'{"outer":[{"metric":"\\ud800"}]}', "invalid_json", "/outer/0/metric"),
            (b'{"record_id":"\xff"}', "invalid_json", "/"),
            (b'{"record_id":', "invalid_json", "/"),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.json"
            for raw, code, pointer in cases:
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    with self.assertRaises(validator.EvidenceError) as caught:
                        validator.read_json(path)
                    self.assertEqual((caught.exception.code, caught.exception.pointer), (code, pointer))

    def test_cli_reports_rule_pointer_and_record_id(self) -> None:
        record = copy.deepcopy(self.blueprint)
        record["cell"]["matrix"]["rows"] = 3
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            run = subprocess.run([sys.executable, str(SCRIPT), "--schema", str(SCHEMA),
                                  "--input", str(path)], capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("selection_mismatch /cell/matrix/rows", run.stderr)
        self.assertIn("record_id=example-s215-cholmod-symmetric4-blueprint", run.stderr)


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = validator.read_json(SCHEMA)

    def check_failure(self, record: dict, code: str, pointer: str, raw: dict | None = None) -> None:
        with self.assertRaises(validator.EvidenceError) as caught:
            validator.validate_semantics(record, self.schema, raw)
        self.assertEqual((caught.exception.code, caught.exception.pointer), (code, pointer))

    def test_planning_blocked_noncomparable_and_observed_shapes(self) -> None:
        validator.validate_semantics(validator.read_json(BLUEPRINT), self.schema)
        not_run = validator.read_json(BLUEPRINT)
        not_run.update(record_kind="run", evidence_origin="generated_local",
                       evidence_status="not_run", run_id="synthetic-not-run", claim_scope="no_result")
        validator.validate_semantics(not_run, self.schema)
        blocked = validator.read_json(BLUEPRINT)
        blocked.update(record_kind="run", evidence_origin="generated_local", evidence_status="blocked",
                       run_id="synthetic-blocked", block_gate="execution", claim_scope="no_result")
        blocked["correctness"]["baseline"].update(state="blocked", availability_reason="cholmod_missing")
        validator.validate_semantics(blocked, self.schema)
        noncomparable = validator.read_json(BLUEPRINT)
        noncomparable.update(record_kind="run", evidence_origin="generated_local",
                             evidence_status="non_comparable", run_id="synthetic-mismatch",
                             mismatch_field="baseline.loaded_library_sha256", claim_scope="no_result")
        validator.validate_semantics(noncomparable, self.schema)
        validator.validate_semantics(synthetic_observed(), self.schema)

    def test_failed_and_invalid_timing_shapes(self) -> None:
        failed = synthetic_observed()
        failed["evidence_status"] = "correctness_failed"
        failed["correctness"]["baseline"].update(state="fail", availability_reason="oracle_gate_failed")
        failed["correctness"]["baseline"]["solution"] = [0.0, 0.0, 0.0, 0.0]
        failed["correctness"]["baseline"].update(
            validator._solution_metrics(failed["correctness"]["baseline"]["solution"]))
        failed["correctness"]["cross_solver_difference_inf"] = None
        failed["correctness"]["cross_solver_consistency_limit"] = None
        failed["claim_scope"] = "no_result"
        failed["non_claims"] = "no_external_solver_result;" + failed["non_claims"]
        validator.validate_semantics(failed, self.schema)
        invalid = synthetic_observed()
        invalid["evidence_status"] = "timing_invalid"
        invalid["timing"].update(state="invalid", raw_samples_artifact="attempt.json",
                                 availability_reason="resolution_limited")
        invalid["artifacts"].append({"path": "attempt.json", "role": "timing_attempt",
                                     "media_type": "application/json", "sha256": "a" * 64})
        validator.validate_semantics(invalid, self.schema)

    def test_complete_cell_and_provenance_mutations(self) -> None:
        cases = (
            (lambda r: r.__setitem__("cell_id", "sha256:" + "0" * 64),
             "cell_id_mismatch", "/cell_id"),
            (lambda r: r["cell"]["project"].__setitem__("compiler", None),
             "missing_provenance", "/cell/project/compiler"),
            (lambda r: r["cell"]["baseline"]["dependencies"].append(
                r["cell"]["baseline"]["dependencies"][0]),
             "cell_id_mismatch", "/identity/baseline/dependencies/1"),
            (lambda r: r["cell"]["baseline"].__setitem__("loaded_library_path", "/opt/../libcholmod.dylib"),
             "missing_provenance", "/cell/baseline/loaded_library_path"),
            (lambda r: r["cell"]["matrix"].__setitem__("source_commit", "aa1c04ea"),
             "missing_provenance", "/cell/matrix/source_commit"),
            (lambda r: r["methodology"]["identity"].__setitem__("path", "README.md"),
             "selection_mismatch", "/methodology/identity/path"),
            (lambda r: r["environment"].__setitem__("timestamp_utc", "2026-02-30T00:00:00Z"),
             "missing_provenance", "/environment/timestamp_utc"),
            (lambda r: r["correctness"]["project"].__setitem__("residual_inf", 1.0),
             "status_contradiction", "/correctness/project/residual_inf"),
            (lambda r: r["correctness"]["baseline"]["phase_statuses"].__setitem__("factor_status", 0),
             "status_contradiction", "/correctness/baseline/phase_statuses/factor_status"),
        )
        for mutate, code, pointer in cases:
            with self.subTest(pointer=pointer):
                record = synthetic_observed()
                mutate(record)
                self.check_failure(record, code, pointer)

    def test_status_and_claim_mutations(self) -> None:
        record = synthetic_observed()
        record["evidence_status"] = "measured"
        record["correctness"]["baseline"]["state"] = "blocked"
        self.check_failure(record, "measured_requires_two_passes_and_raw_samples", "/correctness")
        blocked = validator.read_json(BLUEPRINT)
        blocked.update(record_kind="run", evidence_origin="generated_local", evidence_status="blocked",
                       run_id="synthetic-blocked", block_gate="execution", claim_scope="no_result")
        blocked["timing"]["state"] = "measured"
        self.check_failure(blocked, "blocked_cannot_claim_performance", "/timing/state")
        bad_claim = synthetic_observed()
        bad_claim["non_claims"] += ";speedup=2"
        self.check_failure(bad_claim, "unearned_claim", "/non_claims")
        approval = synthetic_observed()
        approval["publication_review"] = "approved"
        self.check_failure(approval, "missing_provenance", "/publication_review_artifact")

    def test_measured_summary_recomputed_from_raw_pairs(self) -> None:
        record, raw = synthetic_measured()
        validator.validate_semantics(record, self.schema, raw)
        self.check_failure(record, "measured_requires_two_passes_and_raw_samples",
                           "/timing/raw_samples_artifact")
        wrong_summary = copy.deepcopy(record)
        wrong_summary["timing"]["windows"][0]["project"]["median_ns"] = 1
        self.check_failure(wrong_summary, "unit_or_sample_contract", "/timing/windows/0/project/median_ns", raw)
        wrong_order = copy.deepcopy(raw)
        wrong_order["factor_plus_first_solve"]["timed"][1]["order"] = "AB"
        self.check_failure(record, "unit_or_sample_contract", "/timing/windows/0/timed/1", wrong_order)

    def test_failure_label_requires_observed_failure(self) -> None:
        record = synthetic_observed()
        record["evidence_status"] = "correctness_failed"
        record["correctness"]["baseline"].update(state="fail", availability_reason="failed")
        record["correctness"]["cross_solver_difference_inf"] = None
        record["correctness"]["cross_solver_consistency_limit"] = None
        record["claim_scope"] = "no_result"
        record["non_claims"] = "no_external_solver_result;" + record["non_claims"]
        self.check_failure(record, "status_contradiction", "/correctness/baseline")
        record["correctness"]["baseline"]["phase_statuses"]["factor_status"] = 0
        validator.validate_semantics(record, self.schema)

    def test_boolean_status_and_nonfinite_in_memory_are_rejected(self) -> None:
        record = synthetic_observed()
        record["correctness"]["baseline"]["phase_statuses"]["load_status"] = False
        self.check_failure(record, "exact_type", "/correctness/baseline/phase_statuses/load_status")
        record = synthetic_observed()
        record["correctness"]["project"]["solution"][0] = float("nan")
        self.check_failure(record, "nonfinite_or_missing_value_abuse", "/correctness/project/solution/0")

    def test_publication_block_with_two_passes_requires_identity(self) -> None:
        record = synthetic_observed()
        record.update(evidence_status="blocked", block_gate="publication", cell_id=None,
                      cell_id_reason="missing_identity")
        self.check_failure(record, "missing_provenance", "/cell_id")
        record["cell_id"] = validator.canonical_cell_id(validator.identity_from_record(record))[1]
        record["cell_id_reason"] = None
        validator.validate_semantics(record, self.schema)

    def test_partial_noncomparable_run_does_not_need_complete_identity(self) -> None:
        record = synthetic_observed()
        record.update(evidence_status="non_comparable", mismatch_field="baseline_index_abi",
                      cell_id=None, cell_id_reason="baseline_mismatch", claim_scope="no_result")
        record["correctness"]["baseline"].update(state="blocked", phase_statuses=None,
                                                  factor_diagnostics=None, solution=None,
                                                  residual_inf=None, backward_error=None,
                                                  oracle_error_inf=None, residual_limit=None,
                                                  oracle_limit=None, raw_artifact=None,
                                                  availability_reason="abi_mismatch")
        record["correctness"]["cross_solver_difference_inf"] = None
        record["correctness"]["cross_solver_consistency_limit"] = None
        record["non_claims"] = "no_external_solver_result;" + record["non_claims"]
        validator.validate_semantics(record, self.schema)

    def test_measured_method_and_raw_reference_contract(self) -> None:
        record, raw = synthetic_measured()
        record["methodology"]["timing_policy"] = "not_run"
        self.check_failure(record, "unit_or_sample_contract", "/methodology/timing_policy", raw)
        record, raw = synthetic_measured()
        record["timing"]["windows"][0]["raw_samples_artifact"] = "other.json"
        self.check_failure(record, "unit_or_sample_contract", "/timing/windows/0/raw_samples_artifact", raw)
        record, raw = synthetic_measured()
        for phase in ("warmup", "timed"):
            for pair in raw["factor_plus_first_solve"][phase]:
                pair["order"] = "BA" if pair["order"] == "AB" else "AB"
        validator.validate_semantics(record, self.schema, raw)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = validator.read_json(SCHEMA)

    def check_failure(self, record: dict, bundle: Path | None, code: str, pointer: str) -> None:
        with self.assertRaises(validator.EvidenceError) as caught:
            validator.validate_record(record, self.schema, ROOT, bundle)
        self.assertEqual((caught.exception.code, caught.exception.pointer), (code, pointer))

    def test_planning_blueprint_has_no_observation_or_generated_artifacts(self) -> None:
        record = validator.read_json(BLUEPRINT)
        self.assertEqual(record["evidence_status"], "planned")
        self.assertIsNone(record["run_id"])
        self.assertIsNone(record["cell_id"])
        self.assertEqual([record["correctness"][side]["state"] for side in ("project", "baseline")],
                         ["not_run", "not_run"])
        self.assertEqual(record["timing"]["state"], "not_run")
        self.assertEqual(record["artifacts"], [])
        validator.validate_record(record, self.schema, ROOT)

    def test_all_evidence_statuses_have_integrated_synthetic_fixtures(self) -> None:
        expected = set(self.schema["properties"]["evidence_status"]["enum"])
        fixtures = {
            "planned": lambda: validator.read_json(PLANNED),
            "not_run": synthetic_not_run,
            "blocked": synthetic_execution_blocked,
            "non_comparable": synthetic_non_comparable,
            "correctness_failed": synthetic_correctness_failed,
            "correctness_observed": synthetic_observed,
            "timing_invalid": synthetic_timing_invalid,
            "measured": synthetic_measured,
        }
        self.assertEqual(set(fixtures), expected)
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            for status, build in fixtures.items():
                with self.subTest(status=status):
                    fixture = build()
                    record, raw = fixture if status == "measured" else (fixture, None)
                    if record["artifacts"]:
                        stage_synthetic_bundle(record, bundle, raw)
                    validator.validate_record(record, self.schema, ROOT, bundle)
                    self.assertEqual(record["evidence_status"], status)

    def test_failed_and_invalid_statuses_reject_missing_raw_or_observed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            failed = synthetic_correctness_failed()
            stage_synthetic_bundle(failed, bundle)
            (bundle / "baseline.json").unlink()
            self.check_failure(failed, bundle, "artifact_path_or_hash", "/artifacts/1/path")
            failed = synthetic_correctness_failed()
            failed["correctness"]["baseline"].update(solution=[1.0] * 4,
                **validator._solution_metrics([1.0] * 4))
            stage_synthetic_bundle(failed, bundle)
            self.check_failure(failed, bundle, "status_contradiction", "/correctness/baseline")
            invalid = synthetic_timing_invalid()
            stage_synthetic_bundle(invalid, bundle)
            (bundle / "attempt.json").unlink()
            self.check_failure(invalid, bundle, "artifact_path_or_hash", "/artifacts/2/path")
            stage_synthetic_bundle(invalid, bundle)
            invalid["evidence_status"] = "measured"
            self.check_failure(invalid, bundle, "status_contradiction", "/timing/state")
            mismatch = synthetic_non_comparable()
            mismatch["cell_id"] = "sha256:" + "0" * 64
            mismatch["cell_id_reason"] = None
            self.check_failure(mismatch, bundle, "status_contradiction", "/evidence_status")

    def test_observed_and_measured_synthetic_bundles_use_hashed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            observed = synthetic_observed()
            stage_synthetic_bundle(observed, bundle)
            validator.validate_record(observed, self.schema, ROOT, bundle)
            self.assertEqual(observed["evidence_status"], "correctness_observed")
            self.assertIn("synthetic", observed["record_id"])
            self.assertIn("synthetic", observed["environment"]["os_build"])
            self.assertEqual(observed["cell_id"],
                             validator.canonical_cell_id(validator.identity_from_record(observed))[1])
            for name in validator.METHOD_PATHS:
                method = observed["methodology"][name]
                self.assertEqual(method["sha256"], hashlib.sha256((ROOT / method["path"]).read_bytes()).hexdigest())
            for side_name in ("project", "baseline"):
                side = observed["correctness"][side_name]
                raw = json.loads((bundle / side["raw_artifact"]).read_bytes())
                self.assertEqual(set(raw), validator.RAW_SIDE_KEYS)
                for key in validator.RAW_SIDE_KEYS:
                    self.assertEqual(raw[key], side[key])
                self.assertLessEqual(side["residual_inf"], side["residual_limit"])
                self.assertLessEqual(side["oracle_error_inf"], side["oracle_limit"])
            measured, raw = synthetic_measured()
            stage_synthetic_bundle(measured, bundle, raw)
            validator.validate_record(measured, self.schema, ROOT, bundle)
            self.assertEqual(measured["evidence_status"], "measured")
            self.assertEqual(measured["cell_id"], observed["cell_id"])
            self.assertEqual(measured["environment"]["clock_identity"],
                             measured["methodology"]["planned_clock"])
            self.assertEqual(measured["environment"]["threads_requested"], {"blas": 1, "openmp": 1})
            self.assertEqual(measured["environment"]["threads_observed"], {"blas": 1, "openmp": 1})
            self.assertEqual(set(raw), set(measured["methodology"]["planned_phases"]))
            for window in measured["timing"]["windows"]:
                phase = window["phase"]
                self.assertEqual((window["warmup_pairs_observed"], window["timed_pairs_observed"]), (5, 50))
                self.assertEqual(len(raw[phase]["warmup"]), 5)
                self.assertEqual(len(raw[phase]["timed"]), 50)
                for side in ("project", "baseline"):
                    self.assertEqual(window[side], validator._summary(
                        [pair[f"{side}_ns"] for pair in raw[phase]["timed"]]))
            self.assertEqual(hashlib.sha256((bundle / "samples.json").read_bytes()).hexdigest(),
                             measured["artifacts"][2]["sha256"])
            for artifact in measured["artifacts"]:
                self.assertEqual(artifact["sha256"],
                                 hashlib.sha256((bundle / artifact["path"]).read_bytes()).hexdigest())

    def test_strong_status_negative_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            observed = synthetic_observed()
            stage_synthetic_bundle(observed, bundle)
            one_sided = copy.deepcopy(observed)
            one_sided["correctness"]["baseline"].update(state="blocked", phase_statuses=None,
                factor_diagnostics=None, solution=None, residual_inf=None, backward_error=None,
                oracle_error_inf=None, residual_limit=None, oracle_limit=None, raw_artifact=None,
                availability_reason="synthetic_missing_baseline")
            one_sided["correctness"].update(cross_solver_difference_inf=None,
                                            cross_solver_consistency_limit=None)
            self.check_failure(one_sided, bundle, "status_contradiction", "/correctness")
            (bundle / "baseline.json").unlink()
            self.check_failure(observed, bundle, "artifact_path_or_hash", "/artifacts/1/path")
            stage_synthetic_bundle(observed, bundle)
            raw_path = bundle / "project.json"
            altered = json.loads(raw_path.read_bytes())
            altered["solution"][0] = 2.0
            raw_path.write_text(json.dumps(altered), encoding="utf-8")
            observed["artifacts"][0]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            self.check_failure(observed, bundle, "status_contradiction",
                               "/correctness/project/raw_artifact/solution")
            measured, raw = synthetic_measured()
            stage_synthetic_bundle(measured, bundle, raw)
            measured["timing"]["windows"][0]["project"]["median_ns"] = 1
            self.check_failure(measured, bundle, "unit_or_sample_contract",
                               "/timing/windows/0/project/median_ns")
            measured, raw = synthetic_measured()
            stage_synthetic_bundle(measured, bundle, raw)
            measured["timing"]["windows"][0]["warmup_pairs_observed"] = 4
            self.check_failure(measured, bundle, "selection_mismatch",
                               "/timing/windows/0/warmup_pairs_observed")
            measured, raw = synthetic_measured()
            stage_synthetic_bundle(measured, bundle, raw)
            measured["non_claims"] += ";synthetic_winner_project"
            self.check_failure(measured, bundle, "unearned_claim", "/non_claims")

    def test_measured_clock_and_thread_provenance_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            record, raw = synthetic_measured()
            stage_synthetic_bundle(record, bundle, raw)
            record["environment"]["clock_identity"] = None
            self.check_failure(record, bundle, "missing_provenance", "/environment/clock_identity")
            record["environment"]["clock_identity"] = record["methodology"]["planned_clock"]
            record["environment"]["threads_observed"]["blas"] = 2
            record["cell_id"] = validator.canonical_cell_id(validator.identity_from_record(record))[1]
            self.check_failure(record, bundle, "unit_or_sample_contract", "/environment/threads_observed")

    def test_missing_bundle_file_and_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            record = synthetic_observed()
            stage_synthetic_bundle(record, bundle)
            self.check_failure(record, None, "artifact_path_or_hash", "/bundle_root")
            self.check_failure(record, bundle / "missing", "artifact_path_or_hash", "/bundle_root")
            (bundle / "project.json").unlink()
            self.check_failure(record, bundle, "artifact_path_or_hash", "/artifacts/0/path")
            (bundle / "project.json").write_bytes(b"changed\n")
            self.check_failure(record, bundle, "artifact_path_or_hash", "/artifacts/0/path")

    def test_symlink_escape_and_unsafe_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            bundle.mkdir()
            record = synthetic_observed()
            stage_synthetic_bundle(record, bundle)
            outside = Path(temp) / "outside"
            outside.mkdir()
            (outside / "project.json").write_bytes((bundle / "project.json").read_bytes())
            (bundle / "escape").symlink_to(outside, target_is_directory=True)
            record["artifacts"][0]["path"] = "escape/project.json"
            self.check_failure(record, bundle, "artifact_path_or_hash", "/artifacts/0/path")
            record["artifacts"][0]["path"] = "project.json"
            (bundle / "project.json").unlink()
            (bundle / "project.json").symlink_to(outside / "project.json")
            self.check_failure(record, bundle, "artifact_path_or_hash", "/artifacts/0/path")
            for path in ("../project.json", "/tmp/project.json", "nested//project.json",
                         "./project.json", "C:project.json", "C:\\project.json"):
                with self.subTest(path=path):
                    with self.assertRaises(validator.EvidenceError) as caught:
                        validator._read_hashed_file(bundle, path, None, "/artifact")
                    self.assertEqual((caught.exception.code, caught.exception.pointer),
                                     ("artifact_path_or_hash", "/artifact"))

    def test_duplicate_and_unlisted_artifact_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            record = synthetic_observed()
            stage_synthetic_bundle(record, bundle)
            record["artifacts"].append(copy.deepcopy(record["artifacts"][0]))
            self.check_failure(record, bundle, "artifact_path_or_hash", "/artifacts/2/path")
            record["artifacts"].pop()
            record["publication_review_artifact"] = "unlisted-review.txt"
            self.check_failure(record, bundle, "missing_provenance", "/publication_review_artifact")

    def test_measured_raw_json_is_parsed_only_after_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            record, raw = synthetic_measured()
            stage_synthetic_bundle(record, bundle, raw)
            sample = bundle / "samples.json"
            sample.write_bytes(b'{"a":1,"a":2}')
            self.check_failure(record, bundle, "artifact_path_or_hash", "/artifacts/2/path")
            record["artifacts"][2]["sha256"] = hashlib.sha256(sample.read_bytes()).hexdigest()
            self.check_failure(record, bundle, "duplicate_or_unknown_field",
                               "/timing/raw_samples_artifact/a")

    def test_unavailable_root_and_inaccessible_file_fail_closed(self) -> None:
        record = validator.read_json(BLUEPRINT)
        with tempfile.TemporaryDirectory() as temp:
            self.check_failure(record, Path(temp) / "missing", "artifact_path_or_hash", "/bundle_root")
            file_root = Path(temp) / "file"
            file_root.write_bytes(b"not a directory")
            self.check_failure(record, file_root, "artifact_path_or_hash", "/bundle_root")
            loop = Path(temp) / "loop"
            loop.symlink_to(loop)
            self.check_failure(record, loop, "artifact_path_or_hash", "/bundle_root")
        with mock.patch.object(os, "open", side_effect=PermissionError("denied")):
            self.check_failure(record, None, "artifact_path_or_hash", "/cell/matrix/path")

    def test_repository_input_bytes_are_checked_without_mutating_repository(self) -> None:
        record = validator.read_json(BLUEPRINT)
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            paths = [record["cell"]["matrix"]["path"]]
            paths.extend(record["methodology"][name]["path"] for name in validator.METHOD_PATHS)
            for relative in paths:
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            validator.validate_record(record, self.schema, repo)
            matrix = repo / paths[0]
            matrix.write_bytes(b"changed matrix\n")
            with self.assertRaises(validator.EvidenceError) as caught:
                validator.validate_record(record, self.schema, repo)
            self.assertEqual((caught.exception.code, caught.exception.pointer),
                             ("artifact_path_or_hash", "/cell/matrix/path"))
            matrix.write_bytes((ROOT / paths[0]).read_bytes())
            record["methodology"]["identity"]["sha256"] = "0" * 64
            with self.assertRaises(validator.EvidenceError) as caught:
                validator.validate_record(record, self.schema, repo)
            self.assertEqual((caught.exception.code, caught.exception.pointer),
                             ("artifact_path_or_hash", "/methodology/identity/path"))

    def test_cli_reports_missing_bundle_root_for_generated_record(self) -> None:
        record = synthetic_observed()
        with tempfile.TemporaryDirectory() as temp:
            stage_synthetic_bundle(record, Path(temp))
            path = Path(temp) / "record.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            run = subprocess.run([sys.executable, str(SCRIPT), "--schema", str(SCHEMA),
                                  "--input", str(path)], capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("artifact_path_or_hash /bundle_root", run.stderr)
        self.assertIn("record_id=synthetic-observed-v1", run.stderr)


class NonExecutionFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = validator.read_json(SCHEMA)

    def assert_no_result(self, record: dict) -> None:
        self.assertIsNone(record["cell_id"])
        self.assertIsNotNone(record["cell_id_reason"])
        self.assertIsNone(record["environment"])
        self.assertIsNotNone(record["environment_reason"])
        self.assertIn(record["claim_scope"],
                      ("no_result", "planned_one_cell_local_correctness_methodology_only"))
        self.assertIn("no_external_solver_result", record["non_claims"].split(";"))
        self.assertIn("no_timing_result", record["non_claims"].split(";"))
        self.assertEqual(record["publication_review"], "pending")
        self.assertEqual(record["artifacts"], [])
        for name in ("project", "baseline"):
            side = record["correctness"][name]
            self.assertIn(side["state"], ("not_run", "blocked"))
            self.assertIsNotNone(side["availability_reason"])
            for key in ("phase_statuses", "factor_diagnostics", "solution", "residual_inf",
                        "backward_error", "oracle_error_inf", "residual_limit", "oracle_limit",
                        "raw_artifact"):
                self.assertIsNone(side.get(key))
        self.assertEqual(record["timing"]["state"], "not_run")
        self.assertIsNone(record["timing"]["warmup_pairs_observed"])
        self.assertIsNone(record["timing"]["timed_pairs_observed"])
        self.assertIsNone(record["timing"]["raw_samples_artifact"])
        self.assertIsNone(record["timing"]["median_ns"])
        self.assertFalse(record["timing"].get("windows"))
        self.assertIsNotNone(record["timing"]["availability_reason"])

    def test_frozen_planning_fixture_matches_sprint_215_blueprint(self) -> None:
        record = validator.read_json(PLANNED)
        prior = validator.read_json(BLUEPRINT)
        prior["record_id"] = record["record_id"]
        self.assertEqual(record, prior)
        self.assertEqual(record["record_kind"], "blueprint")
        self.assertEqual(record["evidence_origin"], "planning")
        self.assertEqual(record["evidence_status"], "planned")
        self.assertIsNone(record["run_id"])
        self.assert_no_result(record)
        validator.validate_record(record, self.schema, ROOT)

    def test_synthetic_not_run_and_execution_blocked_are_non_promotable(self) -> None:
        for record in (synthetic_not_run(), synthetic_execution_blocked()):
            with self.subTest(status=record["evidence_status"]):
                self.assertEqual(record["record_kind"], "run")
                self.assertEqual(record["evidence_origin"], "generated_local")
                self.assertIsNotNone(record["run_id"])
                self.assert_no_result(record)
                validator.validate_record(record, self.schema, ROOT)
        blocked = synthetic_execution_blocked()
        self.assertEqual(blocked["block_gate"], "execution")
        self.assertEqual(blocked["correctness"]["baseline"]["state"], "blocked")

    def test_nonexecution_labels_cannot_be_promoted_to_pass_or_timing(self) -> None:
        blocked = synthetic_execution_blocked()
        blocked["correctness"]["baseline"]["state"] = "pass"
        with self.assertRaises(validator.EvidenceError) as caught:
            validator.validate_record(blocked, self.schema, ROOT)
        self.assertEqual((caught.exception.code, caught.exception.pointer),
                         ("missing_provenance", "/correctness/baseline/raw_artifact"))
        not_run = synthetic_not_run()
        not_run["evidence_status"] = "measured"
        with self.assertRaises(validator.EvidenceError) as caught:
            validator.validate_record(not_run, self.schema, ROOT)
        self.assertEqual((caught.exception.code, caught.exception.pointer),
                         ("measured_requires_two_passes_and_raw_samples", "/correctness"))

    def test_nonexecution_mutations_reject_missing_reasons_values_and_claims(self) -> None:
        cases = (
            (lambda r: r["correctness"]["baseline"].__setitem__("availability_reason", None),
             "status_contradiction", "/correctness/baseline/availability_reason"),
            (lambda r: r["correctness"]["project"].__setitem__("residual_inf", 0.0),
             "status_contradiction", "/correctness/project"),
            (lambda r: r["timing"].__setitem__("median_ns", {"project": 0, "baseline": 0}),
             "status_contradiction", "/timing"),
            (lambda r: r.__setitem__("cell_id_reason", None),
             "missing_provenance", "/cell_id"),
            (lambda r: r.__setitem__("claim_scope", "one_cell_local_correctness_only"),
             "unearned_claim", "/claim_scope"),
        )
        for mutate, code, pointer in cases:
            with self.subTest(pointer=pointer):
                record = synthetic_execution_blocked()
                mutate(record)
                with self.assertRaises(validator.EvidenceError) as caught:
                    validator.validate_record(record, self.schema, ROOT)
                self.assertEqual((caught.exception.code, caught.exception.pointer),
                                 (code, pointer))

    def test_blocked_fixture_validates_without_claiming_a_result(self) -> None:
        record = synthetic_execution_blocked()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "blocked.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            run = subprocess.run([sys.executable, str(SCRIPT), "--schema", str(SCHEMA),
                                  "--input", str(path)], capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("record_id=synthetic-s221-execution-blocked", run.stdout)
        self.assertIn("evidence_status=blocked", run.stdout)

    def test_missing_cholmod_runtime_module_review_or_binary_cannot_pass(self) -> None:
        cases = (
            (lambda r: r["cell"]["baseline"].update(loaded_library_path=None,
                                                      loaded_library_sha256=None,
                                                      runtime_reason="cholmod_missing"),
             "/cell/baseline/loaded_library_path"),
            (lambda r: r["cell"]["baseline"].update(module_composition=None,
                                                      runtime_reason="module_review_missing"),
             "/cell/baseline/module_composition"),
            (lambda r: r["cell"]["baseline"].update(module_composition=["unverified"]),
             "/cell/baseline/module_composition"),
            (lambda r: r["cell"]["baseline"].update(loaded_library_sha256=None,
                                                      runtime_reason="binary_not_verified"),
             "/cell/baseline/loaded_library_sha256"),
        )
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            for mutate, pointer in cases:
                with self.subTest(pointer=pointer):
                    record = synthetic_observed()
                    mutate(record)
                    stage_synthetic_bundle(record, bundle)
                    with self.assertRaises(validator.EvidenceError) as caught:
                        validator.validate_record(record, self.schema, ROOT, bundle)
                    self.assertEqual((caught.exception.code, caught.exception.pointer),
                                     ("missing_provenance", pointer))

    def test_publication_block_retains_only_hashed_local_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            record = synthetic_observed()
            record.update(evidence_status="blocked", block_gate="publication",
                          status_reason="attribution_review_pending")
            stage_synthetic_bundle(record, bundle)
            self.assertEqual(record["publication_review"], "pending")
            self.assertEqual(record["cell"]["matrix"]["attribution_review"], "pending")
            self.assertEqual(record["claim_scope"], "one_cell_local_correctness_only")
            self.assertIn("no_timing_result", record["non_claims"].split(";"))
            validator.validate_record(record, self.schema, ROOT, bundle)
            record["timing"]["state"] = "measured"
            with self.assertRaises(validator.EvidenceError) as caught:
                validator.validate_semantics(record, self.schema)
            self.assertEqual((caught.exception.code, caught.exception.pointer),
                             ("blocked_cannot_claim_performance", "/timing/state"))


class CliMutationTests(unittest.TestCase):
    def run_cli(self, path: Path, bundle: Path | None = None) -> subprocess.CompletedProcess:
        argv = [sys.executable, str(SCRIPT), "--schema", str(SCHEMA), "--input", str(path)]
        if bundle is not None:
            argv.extend(("--bundle-root", str(bundle)))
        return subprocess.run(argv, capture_output=True, text=True, check=False)

    def assert_accepts(self, path: Path, bundle: Path | None = None) -> None:
        run = self.run_cli(path, bundle)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("external-baseline-evidence-v1: valid", run.stdout)

    def assert_rejects(self, path: Path, code: str, pointer: str,
                       bundle: Path | None = None) -> None:
        run = self.run_cli(path, bundle)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn(f"{code} {pointer} record_id=", run.stderr)
        self.assertNotIn("external-baseline-evidence-v1: valid", run.stdout)

    def assert_semantic_mutation(self, fixture: dict | tuple[dict, dict], mutate,
                                 code: str, pointer: str, restage: bool = False) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            record, raw = fixture if isinstance(fixture, tuple) else (fixture, None)
            if record["evidence_status"] not in ("planned", "not_run", "blocked"):
                stage_synthetic_bundle(record, bundle, raw)
            original = json.dumps(record, sort_keys=True, allow_nan=False)
            original_bytes = {artifact["path"]: (bundle / artifact["path"]).read_bytes()
                              for artifact in record["artifacts"]}
            path = root / "record.json"
            path.write_text(original, encoding="utf-8")
            self.assert_accepts(path, bundle)
            mutate(record, bundle)
            if restage:
                stage_synthetic_bundle(record, bundle, raw)
            path.write_text(json.dumps(record, sort_keys=True, allow_nan=False), encoding="utf-8")
            self.assertEqual(validator.read_json(path), record)
            self.assert_rejects(path, code, pointer, bundle)
            for name, content in original_bytes.items():
                (bundle / name).write_bytes(content)
            path.write_text(original, encoding="utf-8")
            self.assert_accepts(path, bundle)

    def test_semantic_mutations_reject_and_restore(self) -> None:
        def one_side_unobserved(record: dict, _bundle: Path) -> None:
            side = record["correctness"]["baseline"]
            for key in ("phase_statuses", "factor_diagnostics", "solution", "residual_inf",
                        "backward_error", "oracle_error_inf", "residual_limit", "oracle_limit",
                        "raw_artifact"):
                side[key] = None
            side.update(state="not_run", availability_reason="synthetic_side_not_run")

        def one_raw_sample_missing(record: dict, bundle: Path) -> None:
            path = bundle / record["timing"]["raw_samples_artifact"]
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["factor_plus_first_solve"]["timed"].pop()
            content = json.dumps(raw, sort_keys=True).encode("utf-8")
            path.write_bytes(content)
            artifact = next(item for item in record["artifacts"] if item["path"] == path.name)
            artifact["sha256"] = hashlib.sha256(content).hexdigest()

        cases = (
            ("matrix_hash", lambda: synthetic_observed(),
             lambda r, _: r["cell"]["matrix"].__setitem__("sha256", "0" * 64),
             "selection_mismatch", "/cell/matrix/sha256", False),
            ("artifact_bytes", lambda: synthetic_observed(),
             lambda r, b: (b / r["artifacts"][0]["path"]).write_bytes(b"corrupt\n"),
             "artifact_path_or_hash", "/artifacts/0/path", False),
            ("artifact_hash", lambda: synthetic_observed(),
             lambda r, _: r["artifacts"][0].__setitem__("sha256", "0" * 64),
             "artifact_path_or_hash", "/artifacts/0/path", False),
            ("unsafe_artifact_path", lambda: synthetic_observed(),
             lambda r, _: r["artifacts"][0].__setitem__("path", "./project.json"),
             "artifact_path_or_hash", "/artifacts/0/path", False),
            ("canonical_id", lambda: synthetic_observed(),
             lambda r, _: r.__setitem__("cell_id", "sha256:" + "0" * 64),
             "cell_id_mismatch", "/cell_id", False),
            ("solver_control", lambda: synthetic_observed(),
             lambda r, _: r["correctness"]["baseline"]["factor_diagnostics"].__setitem__("ordering", 1),
             "status_contradiction", "/correctness/baseline/factor_diagnostics/ordering", True),
            ("dependency_identity", lambda: synthetic_observed(),
             lambda r, _: r["cell"]["baseline"]["dependencies"][0].__setitem__("sha256", "0" * 64),
             "cell_id_mismatch", "/cell_id", False),
            ("planned_metrics", lambda: validator.read_json(PLANNED),
             lambda r, _: r["correctness"]["project"].__setitem__("residual_inf", 0.0),
             "status_contradiction", "/correctness/project", False),
            ("blocked_pass", lambda: synthetic_execution_blocked(),
             lambda r, _: r["correctness"]["baseline"].__setitem__("state", "pass"),
             "missing_provenance", "/correctness/baseline/raw_artifact", False),
            ("observed_one_side", lambda: synthetic_observed(),
             one_side_unobserved, "status_contradiction", "/correctness", False),
            ("measured_no_raw_reference", synthetic_measured,
             lambda r, _: r["timing"].__setitem__("raw_samples_artifact", None),
             "measured_requires_two_passes_and_raw_samples", "/timing/raw_samples_artifact", False),
            ("timing_unit", synthetic_measured,
             lambda r, _: r["timing"].__setitem__("unit", "ms"),
             "unit_or_sample_contract", "/timing/unit", False),
            ("sample_count", synthetic_measured,
             one_raw_sample_missing, "unit_or_sample_contract", "/timing/windows/0", False),
            ("median_aggregate", synthetic_measured,
             lambda r, _: r["timing"]["windows"][0]["project"].__setitem__("median_ns", 1),
             "unit_or_sample_contract", "/timing/windows/0/project/median_ns", False),
            ("false_performance_claim", synthetic_measured,
             lambda r, _: r.__setitem__("non_claims", r["non_claims"] + ";speedup=2"),
             "unearned_claim", "/non_claims", False),
            ("unreviewed_publication", lambda: synthetic_observed(),
             lambda r, _: r.__setitem__("publication_review", "approved"),
             "missing_provenance", "/publication_review_artifact", False),
        )
        for label, fixture, mutate, code, pointer, restage in cases:
            with self.subTest(label=label):
                self.assert_semantic_mutation(fixture(), mutate, code, pointer, restage)

    def test_mutation_closure_cross_field_rules(self) -> None:
        def change_hashed_raw_output(record: dict, bundle: Path) -> None:
            artifact = next(item for item in record["artifacts"] if item["role"] == "project_raw")
            path = bundle / artifact["path"]
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["solution"][0] = 2.0
            content = json.dumps(raw, sort_keys=True).encode("utf-8")
            path.write_bytes(content)
            artifact["sha256"] = hashlib.sha256(content).hexdigest()

        def claim_blocked_timing(record: dict, _bundle: Path) -> None:
            record.update(evidence_status="blocked", block_gate="execution",
                          status_reason="synthetic_execution_blocked")

        def approve_without_listed_review(record: dict, _bundle: Path) -> None:
            record["publication_review"] = "approved"
            record["publication_review_artifact"] = "review.json"
            record["cell"]["matrix"]["attribution_review"] = "approved"

        cases = (
            ("method_hash", lambda: synthetic_observed(),
             lambda r, _: r["methodology"]["identity"].__setitem__("sha256", "0" * 64),
             "artifact_path_or_hash", "/methodology/identity/path"),
            ("rehashed_raw_output_mismatch", lambda: synthetic_observed(),
             change_hashed_raw_output,
             "status_contradiction", "/correctness/project/raw_artifact/solution"),
            ("blocked_measured", synthetic_measured, claim_blocked_timing,
             "blocked_cannot_claim_performance", "/timing/state"),
            ("duplicate_dependency", lambda: synthetic_observed(),
             lambda r, _: r["cell"]["baseline"]["dependencies"].append(
                 copy.deepcopy(r["cell"]["baseline"]["dependencies"][0])),
             "cell_id_mismatch", "/identity/baseline/dependencies/1"),
            ("unlisted_publication_review", lambda: synthetic_observed(),
             approve_without_listed_review,
             "missing_provenance", "/publication_review_artifact"),
        )
        for label, fixture, mutate, code, pointer in cases:
            with self.subTest(label=label):
                self.assert_semantic_mutation(fixture(), mutate, code, pointer)

    def test_copied_planning_fixture_parser_and_shape_mutations(self) -> None:
        def duplicate_key(path: Path) -> None:
            original = path.read_bytes()
            path.write_bytes(original.replace(b'"record_id":',
                                              b'"record_id":"duplicate","record_id":', 1))

        def unknown_field(path: Path) -> None:
            record = validator.read_json(path)
            record["synthetic_unknown"] = "not_selected"
            path.write_text(json.dumps(record), encoding="utf-8")

        def nonfinite_value(path: Path) -> None:
            record = validator.read_json(path)
            record["timing"]["median_ns"] = float("nan")
            path.write_text(json.dumps(record, allow_nan=True), encoding="utf-8")

        cases = (("duplicate_key", duplicate_key, "duplicate_or_unknown_field", "/record_id"),
                 ("unknown_field", unknown_field, "duplicate_or_unknown_field", "/synthetic_unknown"),
                 ("nonfinite_value", nonfinite_value, "nonfinite_or_missing_value_abuse",
                  "/timing/median_ns"))
        for label, mutate, code, pointer in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "planned.json"
                shutil.copyfile(PLANNED, path)
                self.assert_accepts(path)
                mutate(path)
                self.assert_rejects(path, code, pointer)
                shutil.copyfile(PLANNED, path)
                self.assert_accepts(path)

    def test_missing_execution_provenance_rejects_hashed_observed_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            record = synthetic_observed()
            stage_synthetic_bundle(record, bundle)
            path = root / "observed.json"
            original = json.dumps(record, sort_keys=True)
            path.write_text(original, encoding="utf-8")
            self.assert_accepts(path, bundle)
            record["cell"]["baseline"]["loaded_library_sha256"] = None
            path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            self.assert_rejects(path, "missing_provenance", "/cell/baseline/loaded_library_sha256",
                                bundle)
            path.write_text(original, encoding="utf-8")
            self.assert_accepts(path, bundle)


if __name__ == "__main__":
    unittest.main()
