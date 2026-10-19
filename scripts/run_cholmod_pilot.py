#!/usr/bin/env python3
"""Opt-in, correctness-only CHOLMOD pilot attempt; never publishes evidence."""

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests/data/symmetric_4.mtx"
FIXTURE_SHA256 = "ead030a6d92a4c3197409aba1f4addb9d1e78f3696d52e072686d762705a7335"
LIB_SHA256 = "0f99009a37904313dd3b1c05d30fa3d43ab7cfe8b3a7641b71ac4b725ae8710c"
SELECTION_ID = "s215-cholmod-v7_12_2-symmetric4-spdsolve-v1"
EVIDENCE_SCHEMA = ROOT / "tests/corpus/schemas/external_baseline_evidence_v1.json"
EVIDENCE_VALIDATOR = ROOT / "scripts/validate_external_baseline_evidence.py"
MATRIX = ((10.0, 1.0, 2.0, 0.0), (1.0, 12.0, 1.0, 0.0),
          (2.0, 1.0, 14.0, 0.0), (0.0, 0.0, 0.0, 16.0))
RHS = (13.0, 14.0, 17.0, 16.0)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "SPARSE_IDX_BITS", "CFLAGS")


class PilotError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_side_output(raw: str, expected_side: str) -> dict:
    fields = {}
    for line in raw.splitlines():
        if "=" not in line:
            raise PilotError("malformed_output", f"non-field line: {line!r}")
        key, value = line.split("=", 1)
        if not key.isidentifier() or key in fields:
            raise PilotError("malformed_output", f"invalid or duplicate field: {key!r}")
        fields[key] = value
    if fields.get("side") != expected_side or fields.get("load_status") != "0":
        raise PilotError("identity_or_load_failure", "side or fixture load status mismatch")
    if fields.get("matrix_nnz") != "10" or "loaded_cholmod_path" not in fields:
        raise PilotError("identity_or_load_failure", "missing matrix or loaded-library identity")
    if expected_side == "project":
        required = {"factor_status": "0", "solve_status": "0", "used_csc_path": "0"}
    else:
        required = {
            "csc_nnz": "7", "csc_stype": "-1", "csc_sorted": "1", "csc_packed": "1",
            "csc_itype": "0", "analyze_status": "0", "factor_status": "1",
            "cholmod_status": "0", "factor_minor": "4", "ordering": "0",
            "is_super": "0", "solve_status": "1", "cholmod_solve_status": "0",
        }
    allowed = {"loaded_cholmod_path", "side", "load_status", "matrix_nnz", "x", *required}
    if expected_side == "baseline":
        allowed.add("is_ll")
    if fields.keys() - allowed:
        raise PilotError("malformed_output", f"unexpected fields: {sorted(fields.keys() - allowed)}")
    for key, value in required.items():
        if fields.get(key) != value:
            raise PilotError("solver_status_failure", f"{expected_side}.{key} != {value}")
    try:
        solution = tuple(float(item) for item in fields["x"].split(","))
    except (KeyError, ValueError) as exc:
        raise PilotError("malformed_output", "missing or invalid solution vector") from exc
    if len(solution) != 4 or not all(math.isfinite(value) for value in solution):
        raise PilotError("nonfinite_or_wrong_solution", "solution must have four finite values")
    return {"fields": fields, "solution": solution}


def check_correctness(solution: tuple[float, ...]) -> dict:
    residuals = [sum(MATRIX[row][col] * solution[col] for col in range(4)) - RHS[row]
                 for row in range(4)]
    if not all(math.isfinite(value) for value in residuals):
        raise PilotError("nonfinite_metric", "residual is nonfinite")
    residual_inf = max(abs(value) for value in residuals)
    oracle_error_inf = max(abs(value - 1.0) for value in solution)
    x_inf = max(abs(value) for value in solution)
    denominator = 17.0 * x_inf + 17.0
    if not math.isfinite(denominator):
        raise PilotError("nonfinite_metric", "residual denominator is nonfinite")
    t = 32.0 * 4.0 * sys.float_info.epsilon
    residual_limit = t * 17.0 + t * denominator
    oracle_limit = 2.0 * t * (17.0 / 7.0)
    return {
        "residual_inf": residual_inf,
        "backward_error": residual_inf / denominator,
        "oracle_error_inf": oracle_error_inf,
        "residual_limit": residual_limit,
        "oracle_limit": oracle_limit,
        "pass": residual_inf <= residual_limit and oracle_error_inf <= oracle_limit,
    }


def command(bundle: Path, label: str, argv: list[str], *, cwd: Path = ROOT,
            env: dict | None = None) -> dict:
    failure_kind = None
    try:
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                                timeout=120, check=False)
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        returncode = 124
        failure_kind = "timeout"
    except OSError as exc:
        stdout, stderr, returncode = "", str(exc), 127
        failure_kind = "launch_error"
    files = {}
    for suffix, value in (("stdout", stdout), ("stderr", stderr)):
        path = bundle / f"{label}.{suffix}.txt"
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value, encoding="utf-8")
        files[suffix] = {"path": path.name, "sha256": sha256(path)}
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    entry = {"argv": argv, "cwd": str(cwd), "returncode": returncode,
             "files": files, "stdout": stdout.strip()}
    if failure_kind:
        entry["failure_kind"] = failure_kind
    return entry


def checked_probe(bundle: Path, attempts: dict, label: str, argv: list[str]) -> str:
    entry = command(bundle, label, argv)
    attempts[label] = entry
    if entry["returncode"] != 0:
        raise PilotError("missing_dependency", f"{label} failed with {entry['returncode']}")
    return entry["stdout"]


def run(bundle: Path) -> dict:
    record = {
        "selection_id": SELECTION_ID,
        "record_kind": "unvalidated_attempt",
        "provisional_status": "not_run",
        "timing": "not_run",
        "publication_review": "pending",
        "schema_validation": {"status": "not_run", "reason": "preflight_not_completed"},
        "platform": platform.platform(),
        "environment": {name: os.environ.get(name) for name in THREAD_VARS},
        "commands": {},
        "inputs": {},
        "sides": {},
    }
    attempts = record["commands"]
    try:
        if not FIXTURE.is_file():
            raise PilotError("fixture_missing", "selected fixture is unavailable")
        if sha256(FIXTURE) != FIXTURE_SHA256:
            raise PilotError("fixture_hash_mismatch", "selected fixture bytes changed")
        record["inputs"]["matrix"] = {"path": str(FIXTURE.relative_to(ROOT)), "sha256": FIXTURE_SHA256}
        version = checked_probe(bundle, attempts, "cholmod_version", ["pkg-config", "--modversion", "cholmod"])
        prefix = checked_probe(bundle, attempts, "cholmod_prefix", ["pkg-config", "--variable=prefix", "cholmod"])
        cflags = checked_probe(bundle, attempts, "cholmod_cflags", ["pkg-config", "--cflags", "cholmod"])
        libs = checked_probe(bundle, attempts, "cholmod_libs", ["pkg-config", "--libs", "cholmod"])
        if version != "5.3.4" or Path(prefix).name != "7.12.2":
            raise PilotError("baseline_version_mismatch", f"CHOLMOD {version}, prefix {prefix}")
        library = Path(prefix) / "lib/libcholmod.5.3.4.dylib"
        if not library.is_file() or sha256(library) != LIB_SHA256:
            raise PilotError("baseline_binary_mismatch", "candidate CHOLMOD dylib is absent or changed")
        record["inputs"]["candidate_library"] = {"path": str(library), "sha256": LIB_SHA256}
        record["inputs"]["package"] = {"cholmod_version": version, "prefix": prefix,
                                          "cflags": cflags, "libs": libs}
        record["inputs"]["git_commit"] = checked_probe(bundle, attempts, "git_commit", ["git", "rev-parse", "HEAD"])
        record["inputs"]["git_status"] = checked_probe(bundle, attempts, "git_status", ["git", "status", "--porcelain"])
        record["inputs"]["compiler"] = checked_probe(bundle, attempts, "compiler_version", ["cc", "--version"])
        checked_probe(bundle, attempts, "library_linkage", ["otool", "-L", str(library)])
        if not EVIDENCE_SCHEMA.is_file() or not EVIDENCE_VALIDATOR.is_file():
            record["schema_validation"]["reason"] = "v1_schema_or_validator_missing"
            raise PilotError("schema_validation_unavailable", "v1 schema and validator are required before solving")
        try:
            checked_probe(bundle, attempts, "schema_preflight", [
                sys.executable, str(EVIDENCE_VALIDATOR), "--schema", str(EVIDENCE_SCHEMA), "--self-test",
            ])
        except PilotError as exc:
            record["schema_validation"] = {"status": "failed", "reason": "validator_self_test_failed"}
            raise PilotError("schema_validation_failed", str(exc)) from exc
        record["schema_validation"] = {"status": "preflight_pass", "reason": "negative_fixtures_checked"}
        checked_probe(bundle, attempts, "project_library_build", ["make", "build/libsparse_lu_ortho.a"])
        project_archive = ROOT / "build/libsparse_lu_ortho.a"
        record["inputs"]["project_archive"] = {"path": str(project_archive), "sha256": sha256(project_archive)}
        binary = bundle / "cholmod_pilot"
        compile_argv = [
            "cc", "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Wshadow", "-Wconversion", "-Werror", "-O2",
            "-Iinclude", "-Ibuild/include", *shlex.split(cflags), "scripts/cholmod_pilot.c",
            "-Lbuild", "-lsparse_lu_ortho", *shlex.split(libs), "-lm", "-o", str(binary),
        ]
        checked_probe(bundle, attempts, "compile_helper", compile_argv)
        record["inputs"]["helper_binary"] = {"path": str(binary), "sha256": sha256(binary)}
        for side in ("project", "baseline"):
            entry = command(bundle, side, [str(binary), side, str(FIXTURE)])
            attempts[side] = entry
            if entry["returncode"] != 0:
                raise PilotError("solver_execution_failure", f"{side} exited {entry['returncode']}")
            parsed = parse_side_output(entry["stdout"], side)
            loaded = Path(parsed["fields"]["loaded_cholmod_path"]).resolve()
            if not loaded.is_file() or sha256(loaded) != LIB_SHA256:
                raise PilotError("loaded_library_mismatch", f"{side} loaded {loaded}")
            metrics = check_correctness(parsed["solution"])
            record["sides"][side] = {
                "raw_stdout": entry["files"]["stdout"], "solution": parsed["solution"],
                "metrics": metrics, "loaded_cholmod_path": str(loaded), "loaded_cholmod_sha256": LIB_SHA256,
            }
            if not metrics["pass"]:
                raise PilotError("correctness_failure", f"{side} failed Day 4 limits")
        record["provisional_status"] = "two_sided_correctness_candidate"
        record["status_reason"] = "raw_outputs_require_v1_validation_and_full_provenance"
    except PilotError as exc:
        record["provisional_status"] = "blocked" if exc.code in {
            "missing_dependency", "baseline_version_mismatch", "baseline_binary_mismatch",
            "loaded_library_mismatch", "fixture_hash_mismatch", "fixture_missing",
            "schema_validation_unavailable", "schema_validation_failed",
        } else "failed"
        record["status_reason"] = exc.code
        record["status_detail"] = str(exc)
    except OSError as exc:
        record["provisional_status"] = "blocked"
        record["status_reason"] = "local_io_failure"
        record["status_detail"] = str(exc)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true", required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    if args.fixture.resolve() != FIXTURE or args.output.resolve() == ROOT:
        parser.error("only the selected fixture and a dedicated output directory are allowed")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "build/external-baseline" / SELECTION_ID):
        parser.error("output must be under the selected build/external-baseline directory")
    output.mkdir(parents=True, exist_ok=False)
    record = run(output)
    (output / "attempt.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print(f"{record['provisional_status']}: {record['status_reason']} ({output / 'attempt.json'})")
    return 0 if record["provisional_status"] == "two_sided_correctness_candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
