#!/usr/bin/env python3
"""V1 shape, identity, semantic and artifact checks for external evidence."""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = "external_baseline_evidence_v1"
IDENTITY_KEYS = frozenset(("matrix_sha256", "problem", "project", "baseline", "threads", "host"))
PROBLEM_KEYS = frozenset(("class", "operation", "rhs", "precision", "project_index_abi",
                          "baseline_index_abi"))
PROJECT_KEYS = frozenset(("git_commit", "worktree_state", "patch_sha256", "binary_sha256",
                          "compiler", "compile_flags", "build_mode", "solver_api", "backend", "ordering"))
BASELINE_KEYS = frozenset(("source_commit", "loaded_library_sha256", "dependencies",
                           "build_options", "module_composition", "solver_api", "backend",
                           "ordering", "postorder"))
HOST_KEYS = frozenset(("os", "os_build", "architecture", "cpu"))
THREAD_KEYS = frozenset(("requested", "observed"))
THREAD_COUNT_KEYS = frozenset(("blas", "openmp"))
DEPENDENCY_KEYS = frozenset(("role", "loaded_path", "sha256"))
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
REQUIRED_NON_CLAIMS = frozenset(("no_portable_speed", "no_hosted_threshold",
                                 "no_cross_platform_timing_parity", "no_state_of_the_art_claim"))
OPTIONAL_NON_CLAIMS = frozenset(("no_external_solver_result", "no_timing_result"))
MATRIX = ((10.0, 1.0, 2.0, 0.0), (1.0, 12.0, 1.0, 0.0),
          (2.0, 1.0, 14.0, 0.0), (0.0, 0.0, 0.0, 16.0))
RHS = (13.0, 14.0, 17.0, 16.0)
MAX_STRUCTURED_RAW_BYTES = 4 * 1024 * 1024
RAW_SIDE_KEYS = frozenset(("phase_statuses", "factor_diagnostics", "solution"))
METHOD_PATHS = {
    "identity": "docs/planning/EPIC_19/SPRINT_215/artifacts/day3-methodology-inputs.md",
    "correctness": "docs/planning/EPIC_19/SPRINT_215/artifacts/day4-correctness-and-tolerance-contract.md",
    "measurement": "docs/planning/EPIC_19/SPRINT_215/artifacts/day5-external-comparison-methodology.md",
}


class EvidenceError(ValueError):
    def __init__(self, code: str, pointer: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.pointer = pointer


def _exact_keys(value: object, keys: frozenset[str], pointer: str) -> dict:
    if type(value) is not dict or value.keys() != keys:
        raise EvidenceError("cell_id_mismatch", pointer, f"expected exactly {sorted(keys)}")
    return value


def _complete(value: object, pointer: str) -> None:
    if value is None:
        raise EvidenceError("missing_provenance", pointer, "complete cell identity cannot contain null")
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise EvidenceError("exact_type", pointer, "identity keys must be strings")
            _complete(child, f"{pointer}/{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _complete(child, f"{pointer}/{index}")
    elif type(value) is str:
        if not value:
            raise EvidenceError("missing_provenance", pointer, "empty identity string")
    elif type(value) is int:
        pass
    elif type(value) is not bool:
        raise EvidenceError("exact_type", pointer, "identity values must be strings, integers or booleans")


def canonical_cell_id(identity: dict) -> tuple[bytes, str]:
    """Return canonical UTF-8 bytes and ID for a complete v1 identity object."""
    value = _exact_keys(identity, IDENTITY_KEYS, "/identity")
    _exact_keys(value["problem"], PROBLEM_KEYS, "/identity/problem")
    project = _exact_keys(value["project"], PROJECT_KEYS, "/identity/project")
    baseline = _exact_keys(value["baseline"], BASELINE_KEYS, "/identity/baseline")
    _exact_keys(value["host"], HOST_KEYS, "/identity/host")
    for pointer, digest in (("/identity/matrix_sha256", value["matrix_sha256"]),
                            ("/identity/project/patch_sha256", project["patch_sha256"]),
                            ("/identity/project/binary_sha256", project["binary_sha256"]),
                            ("/identity/baseline/loaded_library_sha256", baseline["loaded_library_sha256"])):
        if type(digest) is not str or not SHA256_RE.fullmatch(digest):
            raise EvidenceError("missing_provenance", pointer, "expected lowercase SHA-256")
    for pointer, commit in (("/identity/project/git_commit", project["git_commit"]),
                            ("/identity/baseline/source_commit", baseline["source_commit"])):
        if type(commit) is not str or not COMMIT_RE.fullmatch(commit):
            raise EvidenceError("missing_provenance", pointer, "expected complete lowercase Git commit")
    if project["worktree_state"] not in ("clean", "dirty"):
        raise EvidenceError("unsupported_enum", "/identity/project/worktree_state", "expected clean or dirty")
    if type(baseline["postorder"]) is not bool:
        raise EvidenceError("exact_type", "/identity/baseline/postorder", "expected boolean")
    threads = _exact_keys(value["threads"], THREAD_KEYS, "/identity/threads")
    for side in THREAD_KEYS:
        counts = _exact_keys(threads[side], THREAD_COUNT_KEYS, f"/identity/threads/{side}")
        if any(type(count) is not int or count < 1 for count in counts.values()):
            raise EvidenceError("exact_type", f"/identity/threads/{side}", "thread counts must be positive integers")
    dependencies = baseline["dependencies"]
    if type(dependencies) is not list:
        raise EvidenceError("exact_type", "/identity/baseline/dependencies", "expected dependency list")
    ordered = []
    seen_roles = set()
    seen_paths = set()
    for index, dependency in enumerate(dependencies):
        item = _exact_keys(dependency, DEPENDENCY_KEYS, f"/identity/baseline/dependencies/{index}")
        path = item["loaded_path"]
        if (type(path) is not str or not Path(path).is_absolute() or
                ".." in Path(path).parts or str(Path(path)) != path):
            raise EvidenceError("missing_provenance", f"/identity/baseline/dependencies/{index}/loaded_path",
                                "dependency loaded path must be absolute and normalized")
        role = item["role"]
        if type(role) is not str or not role:
            raise EvidenceError("missing_provenance", f"/identity/baseline/dependencies/{index}/role",
                                "dependency role must be a nonempty string")
        if type(item["sha256"]) is not str or not SHA256_RE.fullmatch(item["sha256"]):
            raise EvidenceError("missing_provenance", f"/identity/baseline/dependencies/{index}/sha256",
                                "expected lowercase SHA-256")
        if role in seen_roles or path in seen_paths:
            raise EvidenceError("cell_id_mismatch", f"/identity/baseline/dependencies/{index}",
                                "duplicate dependency role or loaded path")
        seen_roles.add(role)
        seen_paths.add(path)
        ordered.append(item)
    value = dict(value)
    value["baseline"] = dict(baseline)
    value["baseline"]["dependencies"] = sorted(ordered, key=lambda item: (item["role"], item["loaded_path"]))
    _complete(value, "/identity")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    return payload, "sha256:" + hashlib.sha256(payload).hexdigest()


def identity_from_record(record: dict) -> dict:
    """Extract only immutable cell controls; methodology and outputs are run provenance."""
    cell = record["cell"]
    problem = cell["problem"]
    project = cell["project"]
    baseline = cell["baseline"]
    environment = record["environment"]
    return {
        "matrix_sha256": cell["matrix"]["sha256"],
        "problem": {
            "class": problem["class"], "operation": problem["operation"],
            "rhs": cell["rhs"]["values"], "precision": problem["precision"],
            "project_index_abi": problem["project_index_abi"],
            "baseline_index_abi": problem["baseline_index_abi"],
        },
        "project": {key: project[key] for key in PROJECT_KEYS},
        "baseline": {key: baseline[key] for key in BASELINE_KEYS},
        "threads": {"requested": environment["threads_requested"],
                    "observed": environment["threads_observed"]},
        "host": {key: environment[key] for key in HOST_KEYS},
    }


class _Pairs(list):
    """Retain object pairs until nested JSON pointers can be assigned."""


class _Nonfinite:
    def __init__(self, token: str):
        self.token = token


def _pointer(parent: str, key: str) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}" if parent else f"/{escaped}"


def _build_json(value: object, pointer: str) -> object:
    if type(value) is _Nonfinite:
        raise EvidenceError("nonfinite_or_missing_value_abuse", pointer or "/",
                            f"nonfinite JSON number: {value.token}")
    if type(value) is _Pairs:
        result = {}
        for key, child in value:
            try:
                key.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise EvidenceError("invalid_json", pointer or "/", "invalid Unicode object key") from exc
            child_pointer = _pointer(pointer, key)
            if key in result:
                raise EvidenceError("duplicate_or_unknown_field", child_pointer, "duplicate JSON key")
            result[key] = _build_json(child, child_pointer)
        return result
    if type(value) is list:
        return [_build_json(child, _pointer(pointer, str(index))) for index, child in enumerate(value)]
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise EvidenceError("invalid_json", pointer or "/", "invalid Unicode string") from exc
    return value


def _nonfinite(token: str) -> _Nonfinite:
    return _Nonfinite(token)


def _finite_float(token: str) -> float | _Nonfinite:
    value = float(token)
    if not math.isfinite(value):
        return _nonfinite(token)
    return value


def _parse_json_bytes(data: bytes, pointer: str = "") -> dict:
    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_Pairs,
                            parse_constant=_nonfinite, parse_float=_finite_float)
    except EvidenceError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise EvidenceError("invalid_json", pointer or "/", str(exc)) from exc
    result = _build_json(parsed, pointer)
    if type(result) is not dict:
        raise EvidenceError("exact_type", pointer or "/", "JSON root must be an object")
    return result


def read_json(path: Path) -> dict:
    try:
        return _parse_json_bytes(path.read_bytes())
    except EvidenceError:
        raise
    except OSError as exc:
        raise EvidenceError("invalid_json", "/", str(exc)) from exc


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_strict_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def _matches_type(value: object, kind: str) -> bool:
    return {
        "null": lambda: value is None,
        "string": lambda: type(value) is str,
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in (int, float),
        "boolean": lambda: type(value) is bool,
        "object": lambda: type(value) is dict,
        "array": lambda: type(value) is list,
    }[kind]()


def _const_code(pointer: str) -> str:
    if pointer == "/schema_version":
        return "unsupported_enum"
    if pointer.endswith(("/timing_unit", "/unit", "/planned_warmup_pairs", "/planned_timed_pairs")):
        return "unit_or_sample_contract"
    return "selection_mismatch"


def _check_shape(value: object, node: dict, root: dict, pointer: str) -> None:
    if "$ref" in node:
        ref = node["$ref"]
        if type(ref) is not str or not ref.startswith("#/$defs/"):
            raise EvidenceError("invalid_json", pointer or "/", "unsupported schema reference")
        name = ref.removeprefix("#/$defs/")
        if name not in root.get("$defs", {}):
            raise EvidenceError("invalid_json", pointer or "/", "missing schema definition")
        _check_shape(value, root["$defs"][name], root, pointer)
        return
    if "oneOf" in node:
        errors = []
        matches = 0
        for alternative in node["oneOf"]:
            try:
                _check_shape(value, alternative, root, pointer)
                matches += 1
            except EvidenceError as exc:
                errors.append(exc)
        if matches == 1:
            return
        if matches == 0:
            if not errors:
                raise EvidenceError("invalid_json", pointer or "/", "empty schema alternatives")
            raise errors[-1]
        raise EvidenceError("exact_type", pointer or "/", "ambiguous schema alternatives")
    expected = node.get("type")
    if expected is not None:
        kinds = expected if type(expected) is list else [expected]
        if not any(_matches_type(value, kind) for kind in kinds):
            raise EvidenceError("exact_type", pointer or "/", f"expected {expected}")
    if "const" in node and not _strict_equal(value, node["const"]):
        code = "exact_type" if type(value) is not type(node["const"]) else _const_code(pointer)
        raise EvidenceError(code, pointer or "/", "pinned value differs")
    if "enum" in node and not any(_strict_equal(value, item) for item in node["enum"]):
        raise EvidenceError("unsupported_enum", pointer or "/", "unsupported value")
    if type(value) is str:
        if len(value) < node.get("minLength", 0):
            raise EvidenceError("exact_type", pointer or "/", "empty or short string")
        if "pattern" in node and re.search(node["pattern"], value) is None:
            code = "missing_provenance" if pointer.endswith(("sha256", "commit")) else "exact_type"
            raise EvidenceError(code, pointer or "/", "string does not match v1 syntax")
    if type(value) is float and not math.isfinite(value):
        raise EvidenceError("nonfinite_or_missing_value_abuse", pointer or "/", "nonfinite number")
    if type(value) in (int, float) and "minimum" in node and value < node["minimum"]:
        raise EvidenceError("exact_type", pointer or "/", "value below minimum")
    if type(value) is dict:
        for key in sorted(node.get("required", [])):
            if key not in value:
                raise EvidenceError("missing_required_field", _pointer(pointer, key), "required field missing")
        properties = node.get("properties", {})
        extra = node.get("additionalProperties", True)
        for key in sorted(value):
            child_pointer = _pointer(pointer, key)
            if key in properties:
                _check_shape(value[key], properties[key], root, child_pointer)
            elif extra is False:
                raise EvidenceError("duplicate_or_unknown_field", child_pointer, "unknown field")
            elif type(extra) is dict:
                _check_shape(value[key], extra, root, child_pointer)
        if len(value) < node.get("minProperties", 0):
            raise EvidenceError("missing_required_field", pointer or "/", "too few properties")
    if type(value) is list:
        if len(value) < node.get("minItems", 0) or len(value) > node.get("maxItems", float("inf")):
            raise EvidenceError("exact_type", pointer or "/", "array length outside v1 bounds")
        if "items" in node:
            for index, child in enumerate(value):
                _check_shape(child, node["items"], root, _pointer(pointer, str(index)))


def validate_shape(record: dict, schema: dict) -> None:
    if schema.get("$id") != SCHEMA_VERSION or schema.get("type") != "object":
        raise EvidenceError("unsupported_enum", "/$id", "unsupported evidence schema")
    _check_shape(record, schema, schema, "")


def _require(condition: bool, code: str, pointer: str, detail: str) -> None:
    if not condition:
        raise EvidenceError(code, pointer, detail)


def _metric_matches(actual: object, expected: float, pointer: str) -> None:
    _require(type(actual) in (int, float) and math.isfinite(actual),
             "nonfinite_or_missing_value_abuse", pointer, "finite metric required")
    _require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0),
             "status_contradiction", pointer, "reported metric disagrees with raw solution")


def _solution_metrics(solution: list) -> dict:
    residuals = [sum(MATRIX[row][col] * solution[col] for col in range(4)) - RHS[row]
                 for row in range(4)]
    _require(all(math.isfinite(value) for value in residuals), "nonfinite_or_missing_value_abuse",
             "/correctness", "residual calculation overflowed")
    residual = max(abs(value) for value in residuals)
    error = max(abs(value - 1.0) for value in solution)
    denominator = 17.0 * max(abs(value) for value in solution) + 17.0
    _require(math.isfinite(denominator), "nonfinite_or_missing_value_abuse",
             "/correctness", "backward-error denominator overflowed")
    t = 32.0 * 4.0 * sys.float_info.epsilon
    return {"residual_inf": residual, "backward_error": residual / denominator,
            "oracle_error_inf": error, "residual_limit": t * 17.0 + t * denominator,
            "oracle_limit": 2.0 * t * (17.0 / 7.0)}


def _check_side(side: dict, name: str, artifacts: set[str]) -> None:
    pointer = f"/correctness/{name}"
    state = side["state"]
    observed = ("phase_statuses", "factor_diagnostics", "solution", "residual_inf", "backward_error",
                "oracle_error_inf", "residual_limit", "oracle_limit", "raw_artifact")
    if state in ("not_run", "blocked"):
        _require(all(side.get(key) is None for key in observed), "status_contradiction", pointer,
                 "unrun side carries observed values")
        _require(side["availability_reason"] is not None, "status_contradiction",
                 f"{pointer}/availability_reason", "unrun side needs a reason")
        return
    _require(side.get("raw_artifact") in artifacts, "missing_provenance", f"{pointer}/raw_artifact",
             "observed side needs a listed raw artifact")
    _require(side.get("phase_statuses") is not None, "missing_provenance",
             f"{pointer}/phase_statuses", "observed side needs phase statuses")
    expected_status = ({"load_status": 0, "factor_status": 0, "solve_status": 0}
                       if name == "project" else
                       {"load_status": 0, "analyze_status": 0, "factor_status": 1,
                        "solve_status": 1, "cholmod_status": 0})
    expected_diagnostics = ({"used_csc_path": 0} if name == "project" else
                            {"factor_minor": 4, "ordering": 0, "is_super": 0, "csc_itype": 0})
    if state == "fail":
        _require(side.get("availability_reason") is not None, "status_contradiction",
                 f"{pointer}/availability_reason", "failure needs a reason")
        failed_phase = any(type(side["phase_statuses"].get(key)) is int and
                           side["phase_statuses"][key] != expected
                           for key, expected in expected_status.items())
        diagnostics = side.get("factor_diagnostics")
        failed_control = (type(diagnostics) is dict and
                          any(type(diagnostics.get(key)) is int and
                              diagnostics[key] != expected
                              for key, expected in expected_diagnostics.items()))
        failed_numerics = False
        if side.get("solution") is not None:
            metrics = _solution_metrics(side["solution"])
            for key, expected in metrics.items():
                _metric_matches(side.get(key), expected, f"{pointer}/{key}")
            failed_numerics = (metrics["residual_inf"] > metrics["residual_limit"] or
                               metrics["oracle_error_inf"] > metrics["oracle_limit"])
        else:
            _require(all(side.get(key) is None for key in
                         ("residual_inf", "backward_error", "oracle_error_inf",
                          "residual_limit", "oracle_limit")), "status_contradiction", pointer,
                     "missing solution cannot carry numerical metrics")
        _require(failed_phase or failed_control or failed_numerics, "status_contradiction", pointer,
                 "failure label needs an observed failed phase, control or numerical gate")
        return
    _require(side["availability_reason"] is None, "status_contradiction",
             f"{pointer}/availability_reason", "passing side cannot be unavailable")
    for key, expected in expected_status.items():
        value = side["phase_statuses"].get(key)
        _require(type(value) is int and value == expected, "status_contradiction",
                 f"{pointer}/phase_statuses/{key}", "required success status missing or wrong")
    diagnostics = side.get("factor_diagnostics")
    _require(type(diagnostics) is dict, "missing_provenance", f"{pointer}/factor_diagnostics",
             "passing side needs factor diagnostics")
    for key, expected in expected_diagnostics.items():
        value = diagnostics.get(key)
        _require(type(value) is int and value == expected, "status_contradiction",
                 f"{pointer}/factor_diagnostics/{key}", "effective solver control differs")
    solution = side.get("solution")
    _require(type(solution) is list and len(solution) == 4, "missing_provenance",
             f"{pointer}/solution", "passing side needs a four-value solution")
    metrics = _solution_metrics(solution)
    for key, expected in metrics.items():
        _metric_matches(side.get(key), expected, f"{pointer}/{key}")
    _require(metrics["residual_inf"] <= metrics["residual_limit"] and
             metrics["oracle_error_inf"] <= metrics["oracle_limit"],
             "status_contradiction", pointer, "passing side fails Day 4 numerical gates")


def _check_provenance(record: dict) -> None:
    cell = record["cell"]
    environment = record["environment"]
    _require(type(environment) is dict and record["environment_reason"] is None,
             "missing_provenance", "/environment", "observed run needs captured environment")
    try:
        datetime.fromisoformat(environment["timestamp_utc"].removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise EvidenceError("missing_provenance", "/environment/timestamp_utc",
                            "invalid UTC calendar timestamp") from exc
    if record["evidence_origin"] == "generated_ci":
        for key in ("workflow_id", "workflow_run_id", "workflow_artifact_id"):
            _require(environment.get(key) is not None, "missing_provenance",
                     f"/environment/{key}", "hosted run needs workflow provenance")
    _require(cell["matrix"]["source_commit"] == "aa1c04eab3536031b9a92ee92587260520ce1913",
             "missing_provenance", "/cell/matrix/source_commit", "generated run needs full matrix-origin commit")
    project = cell["project"]
    baseline = cell["baseline"]
    for pointer, value in ((f"/environment/{key}", environment[key]) for key in
                           ("os", "os_build", "architecture", "cpu", "compiler_runtime",
                            "blas_identity", "openmp_identity")):
        _require(value.strip().lower() not in ("unknown", "unverified", "n/a"),
                 "missing_provenance", pointer, "placeholder is not observed provenance")
    for name, obj, keys in (("project", project, ("git_commit", "worktree_state", "patch_sha256",
                                                 "compiler", "compile_flags", "build_mode", "binary_sha256")),
                            ("baseline", baseline, ("loaded_library_path", "loaded_library_sha256",
                                                   "build_options", "module_composition", "dependencies"))):
        for key in keys:
            _require(obj.get(key) is not None, "missing_provenance", f"/cell/{name}/{key}",
                     "observed run needs runtime provenance")
            value = obj[key]
            if type(value) is str:
                _require(value.strip().lower() not in ("unknown", "unverified", "n/a"),
                         "missing_provenance", f"/cell/{name}/{key}",
                         "placeholder is not observed provenance")
        _require(obj["runtime_reason"] is None, "status_contradiction", f"/cell/{name}/runtime_reason",
                 "complete runtime cannot carry unavailable reason")
    _require(baseline["dependencies"], "missing_provenance", "/cell/baseline/dependencies",
             "loaded dependency inventory cannot be empty")
    for pointer, path in (("/cell/baseline/loaded_library_path", baseline["loaded_library_path"]),
                          ("/environment/cwd", environment["cwd"])):
        _require(Path(path).is_absolute() and ".." not in Path(path).parts and
                 str(Path(path)) == path, "missing_provenance", pointer,
                 "observed path must be absolute and normalized")
    _require(all(value.strip().lower() not in ("unknown", "unverified", "n/a")
                 for value in baseline["module_composition"]), "missing_provenance",
             "/cell/baseline/module_composition", "placeholder module is not provenance")
    if project["worktree_state"] == "clean":
        _require(project["patch_sha256"] == hashlib.sha256(b"").hexdigest(),
                 "missing_provenance", "/cell/project/patch_sha256",
                 "clean worktree must identify the empty patch")
    _require(cell["problem"]["baseline_index_abi"] is not None and
             cell["problem"]["index_abi_reason"] is None, "missing_provenance",
             "/cell/problem/baseline_index_abi", "observed baseline ABI must be verified")
    for name in ("identity", "correctness", "measurement"):
        _require(record["methodology"][name]["sha256"] is not None, "missing_provenance",
                 f"/methodology/{name}/sha256", "observed run needs methodology hash")
    _require(record["methodology"]["hash_reason"] is None, "status_contradiction",
             "/methodology/hash_reason", "complete methodology cannot carry unavailable reason")


def _summary(values: list[int]) -> dict:
    ordered = sorted(values)
    median = statistics.median(ordered)
    q1 = statistics.median(ordered[:25])
    q3 = statistics.median(ordered[25:])
    return {"sample_count": 50, "minimum_ns": ordered[0], "maximum_ns": ordered[-1],
            "median_ns": median, "q1_ns": q1, "q3_ns": q3, "iqr_ns": q3 - q1,
            "mad_ns": statistics.median(abs(value - median) for value in values)}


def _check_timing(record: dict, raw_samples: dict | None) -> None:
    timing = record["timing"]
    state = timing["state"]
    if state == "not_run":
        _require(all(timing.get(key) is None for key in ("warmup_pairs_observed", "timed_pairs_observed",
                                                    "raw_samples_artifact", "median_ns")) and
                 not timing.get("windows") and timing["availability_reason"] is not None,
                 "status_contradiction", "/timing", "unrun timing cannot carry measurements")
        return
    if state == "invalid":
        _require(timing["availability_reason"] is not None and timing.get("median_ns") is None and
                 not timing.get("windows") and timing["raw_samples_artifact"] in
                 {item["path"] for item in record["artifacts"]},
                 "unit_or_sample_contract", "/timing", "invalid timing needs raw attempt, not summary")
        return
    _require(raw_samples is not None, "measured_requires_two_passes_and_raw_samples",
             "/timing/raw_samples_artifact", "measured timing needs verified raw pairs")
    _require(record["methodology"]["timing_policy"] == "eligible", "unit_or_sample_contract",
             "/methodology/timing_policy", "measured timing needs an eligible method")
    _require(timing["availability_reason"] is None and timing["warmup_pairs_observed"] == 5 and
             timing["timed_pairs_observed"] == 50 and timing["raw_samples_artifact"] is not None,
             "unit_or_sample_contract", "/timing", "measured pair counts or raw reference invalid")
    _require(timing["raw_samples_artifact"] in {item["path"] for item in record["artifacts"]},
             "missing_provenance", "/timing/raw_samples_artifact", "raw samples need a listed artifact")
    windows = timing.get("windows")
    _require(type(windows) is list and windows, "unit_or_sample_contract", "/timing/windows",
             "measured timing needs per-phase windows")
    _require(len({window["phase"] for window in windows}) == len(windows),
             "unit_or_sample_contract", "/timing/windows", "duplicate timing phase")
    _require(type(raw_samples) is dict and set(raw_samples) == {window["phase"] for window in windows},
             "unit_or_sample_contract", "/timing/windows", "raw phase set differs")
    _require(record["environment"].get("clock_resolution_ns") is not None and
             record["environment"]["clock_resolution_ns"] > 0,
             "missing_provenance", "/environment/clock_resolution_ns", "measured clock resolution required")
    _require(record["environment"].get("clock_identity") == record["methodology"]["planned_clock"],
             "missing_provenance", "/environment/clock_identity", "measured clock identity differs from method")
    _require(record["environment"]["threads_requested"] == {"blas": 1, "openmp": 1} and
             record["environment"]["threads_observed"] == {"blas": 1, "openmp": 1},
             "unit_or_sample_contract", "/environment/threads_observed",
             "selected timing requires verified single-thread controls")
    for index, window in enumerate(windows):
        pointer = f"/timing/windows/{index}"
        _require(window["raw_samples_artifact"] == timing["raw_samples_artifact"],
                 "unit_or_sample_contract", f"{pointer}/raw_samples_artifact",
                 "window and run must cite the same raw samples")
        _require(window["quartile_convention"] == "median_of_halves", "unit_or_sample_contract",
                 f"{pointer}/quartile_convention", "unsupported quartile convention")
        raw = raw_samples[window["phase"]]
        _require(type(raw) is dict and set(raw) == {"warmup", "timed"} and
                 type(raw["warmup"]) is list and len(raw["warmup"]) == 5 and
                 type(raw["timed"]) is list and len(raw["timed"]) == 50,
                 "unit_or_sample_contract", pointer, "raw pair counts differ")
        for phase, pairs in (("warmup", raw["warmup"]), ("timed", raw["timed"])):
            first_order = pairs[0].get("order") if type(pairs[0]) is dict else None
            _require(first_order in ("AB", "BA"), "unit_or_sample_contract",
                     f"{pointer}/{phase}/0", "invalid initial pair order")
            for pair_index, pair in enumerate(pairs):
                expected_order = first_order if pair_index % 2 == 0 else ("BA" if first_order == "AB" else "AB")
                _require(type(pair) is dict and pair.get("order") == expected_order and
                         all(pair.get(f"{side}_status") == "pass" and
                             type(pair.get(f"{side}_ns")) is int and pair[f"{side}_ns"] > 0
                             for side in ("project", "baseline")),
                         "unit_or_sample_contract", f"{pointer}/{phase}/{pair_index}",
                         "pair order, status or duration invalid")
        for side in ("project", "baseline"):
            summary = _summary([pair[f"{side}_ns"] for pair in raw["timed"]])
            for key, expected in summary.items():
                _require(window[side][key] == expected, "unit_or_sample_contract",
                         f"{pointer}/{side}/{key}", "summary differs from raw samples")
            _require(summary["median_ns"] >= 100 * record["environment"]["clock_resolution_ns"],
                     "unit_or_sample_contract", f"{pointer}/{side}/median_ns", "resolution-limited timing")
    primary = next((window for window in windows if window["phase"] == "factor_plus_first_solve"), None)
    _require(primary is not None and timing["median_ns"] ==
             {side: primary[side]["median_ns"] for side in ("project", "baseline")},
             "unit_or_sample_contract", "/timing/median_ns", "primary median differs from raw window")


def validate_semantics(record: dict, schema: dict, raw_samples: dict | None = None) -> None:
    """Validate v1 relationships against already verified raw samples."""
    validate_shape(record, schema)
    status = record["evidence_status"]
    for name, expected_path in METHOD_PATHS.items():
        _require(record["methodology"][name]["path"] == expected_path, "selection_mismatch",
                 f"/methodology/{name}/path", "methodology reference differs from selected v1 method")
    kind = record["record_kind"]
    origin = record["evidence_origin"]
    _require((kind == "blueprint" and origin == "planning" and status == "planned" and
              record["run_id"] is None) or
             (kind == "run" and origin in ("generated_local", "generated_ci") and
              status != "planned" and record["run_id"] is not None),
             "status_contradiction", "/record_kind", "kind, origin, run ID and status disagree")
    non_claims = record["non_claims"].split(";")
    _require(len(non_claims) == len(set(non_claims)) and
             REQUIRED_NON_CLAIMS.issubset(non_claims) and
             set(non_claims) <= REQUIRED_NON_CLAIMS | OPTIONAL_NON_CLAIMS,
             "unearned_claim", "/non_claims", "unsupported or missing non-claim token")
    if record["publication_review"] == "approved":
        _require(record.get("publication_review_artifact") is not None and
                 record["cell"]["matrix"]["attribution_review"] == "approved",
                 "missing_provenance", "/publication_review_artifact", "approval needs review evidence")
    sides = record["correctness"]
    side_states = (sides["project"]["state"], sides["baseline"]["state"])
    timing = record["timing"]
    if status == "measured":
        _require(side_states == ("pass", "pass"), "measured_requires_two_passes_and_raw_samples",
                 "/correctness", "measured timing needs two correctness passes")
    if status == "blocked":
        _require(timing["state"] != "measured", "blocked_cannot_claim_performance",
                 "/timing/state", "blocked run cannot claim measured timing")
    artifacts = {artifact["path"] for artifact in record["artifacts"]}
    for name in ("project", "baseline"):
        _check_side(sides[name], name, artifacts)
    if status in ("planned", "not_run"):
        _require(side_states == ("not_run", "not_run") and timing["state"] == "not_run" and
                 record["cell_id"] is None and not record["artifacts"],
                 "status_contradiction", "/evidence_status", "unrun record carries observations")
        if status == "planned":
            _require(record["environment"] is None and record["environment_reason"] is not None and
                     all(record["cell"]["project"][key] is None for key in
                         ("git_commit", "binary_sha256", "compiler", "compile_flags", "build_mode")) and
                     all(record["cell"]["baseline"][key] is None for key in
                         ("loaded_library_path", "loaded_library_sha256", "dependencies")) and
                     all(record["methodology"][name]["sha256"] is None for name in
                         ("identity", "correctness", "measurement")),
                     "status_contradiction", "/evidence_status", "planning record carries execution provenance")
    elif status == "blocked":
        gate = record.get("block_gate")
        _require(gate in ("execution", "publication"), "status_contradiction", "/block_gate",
                 "blocked record needs a named gate")
        if gate == "execution":
            _require(all(state in ("not_run", "blocked") for state in side_states) and
                     timing["state"] == "not_run", "blocked_cannot_claim_performance",
                     "/evidence_status", "execution block cannot carry solver or timing result")
        else:
            _require(timing["state"] != "measured", "blocked_cannot_claim_performance",
                     "/timing/state", "publication block cannot claim measured timing")
    elif status == "non_comparable":
        _require(record.get("mismatch_field") is not None and side_states != ("pass", "pass") and
                 timing["state"] != "measured" and record["cell_id"] is None,
                 "status_contradiction", "/evidence_status", "mismatch cannot reuse complete cell or pair")
    elif status == "correctness_failed":
        _require("fail" in side_states and timing["state"] in ("not_run", "invalid"),
                 "status_contradiction", "/evidence_status", "failed correctness needs a failed side")
    elif status in ("correctness_observed", "timing_invalid", "measured"):
        code = ("measured_requires_two_passes_and_raw_samples" if status == "measured" else
                "status_contradiction")
        _require(side_states == ("pass", "pass"), code, "/correctness",
                 "two independent passes required")
        required_timing = {"correctness_observed": "not_run", "timing_invalid": "invalid",
                           "measured": "measured"}[status]
        _require(timing["state"] == required_timing, "status_contradiction", "/timing/state",
                 "timing state contradicts evidence status")
    if side_states == ("pass", "pass"):
        project_x = sides["project"]["solution"]
        baseline_x = sides["baseline"]["solution"]
        difference = max(abs(a - b) for a, b in zip(project_x, baseline_x))
        limit = _solution_metrics(project_x)["oracle_limit"] + _solution_metrics(baseline_x)["oracle_limit"]
        _metric_matches(sides.get("cross_solver_difference_inf"), difference,
                        "/correctness/cross_solver_difference_inf")
        _metric_matches(sides.get("cross_solver_consistency_limit"), limit,
                        "/correctness/cross_solver_consistency_limit")
        _require(difference <= limit, "status_contradiction", "/correctness",
                 "paired solutions fail consistency gate")
    else:
        _require(sides.get("cross_solver_difference_inf") is None and
                 sides.get("cross_solver_consistency_limit") is None,
                 "status_contradiction", "/correctness", "unpaired record carries paired metrics")
    complete_observation = status in ("correctness_failed", "correctness_observed", "timing_invalid",
                                      "measured") or side_states == ("pass", "pass")
    if complete_observation:
        _check_provenance(record)
    if record["cell_id"] is None:
        _require(record["cell_id_reason"] is not None and
                 not complete_observation,
                 "missing_provenance", "/cell_id", "observed or unexplained cell ID is incomplete")
    else:
        _require(record["cell_id_reason"] is None, "status_contradiction", "/cell_id_reason",
                 "complete cell ID cannot have an unavailable reason")
        _check_provenance(record)
        _, expected_id = canonical_cell_id(identity_from_record(record))
        _require(record["cell_id"] == expected_id, "cell_id_mismatch", "/cell_id",
                 "cell ID differs from complete identity")
    if (status in ("planned", "not_run", "non_comparable") or
            (status == "blocked" and side_states != ("pass", "pass"))):
        _require(record["claim_scope"] in ("planned_one_cell_local_correctness_methodology_only", "no_result") and
                 "no_external_solver_result" in non_claims and "no_timing_result" in non_claims,
                 "unearned_claim", "/claim_scope", "unobserved record claims a result")
    elif status == "measured":
        _require(record["claim_scope"] == "one_cell_local_descriptive_latency_only" and
                 "no_external_solver_result" not in non_claims and "no_timing_result" not in non_claims,
                 "unearned_claim", "/claim_scope", "measured claim exceeds one local cell")
    elif side_states == ("pass", "pass"):
        _require(record["claim_scope"] == "one_cell_local_correctness_only" and
                 "no_external_solver_result" not in non_claims and "no_timing_result" in non_claims,
                 "unearned_claim", "/claim_scope", "correctness claim or non-claims disagree")
    else:
        _require(record["claim_scope"] == "no_result" and
                 "no_external_solver_result" in non_claims and "no_timing_result" in non_claims,
                 "unearned_claim", "/claim_scope", "failed run cannot claim success")
    _check_timing(record, raw_samples)


def _read_hashed_file(root: Path, relative: str, expected: str | None,
                      pointer: str, capture: bool = False) -> bytes | None:
    parts = relative.split("/")
    _require(not relative.startswith("/") and "\\" not in relative and "\x00" not in relative and
             re.match(r"^[A-Za-z]:", relative) is None and
             all(part not in ("", ".", "..") for part in parts),
             "artifact_path_or_hash", pointer, "path must be a normalized relative POSIX path")
    _require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
             "artifact_path_or_hash", pointer, "safe artifact access is unavailable on this platform")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = -1
    try:
        physical_root = root.resolve(strict=True)
        fd = os.open(physical_root, flags | os.O_DIRECTORY)
        for index, part in enumerate(parts):
            next_flags = flags | (os.O_DIRECTORY if index < len(parts) - 1 else 0)
            child = os.open(part, next_flags, dir_fd=fd)
            os.close(fd)
            fd = child
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise EvidenceError("artifact_path_or_hash", pointer, "artifact is not a regular file")
        digest = hashlib.sha256()
        data = bytearray() if capture else None
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            while chunk := stream.read(65536):
                digest.update(chunk)
                if data is not None:
                    data.extend(chunk)
                    _require(len(data) <= MAX_STRUCTURED_RAW_BYTES, "artifact_path_or_hash", pointer,
                             "structured raw artifact exceeds v1 size limit")
        if expected is not None:
            _require(digest.hexdigest() == expected, "artifact_path_or_hash", pointer,
                     "artifact SHA-256 differs from bytes")
        return bytes(data) if data is not None else None
    except EvidenceError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise EvidenceError("artifact_path_or_hash", pointer, str(exc)) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def verify_artifacts(record: dict, repo_root: Path, bundle_root: Path | None) -> dict | None:
    """Verify repository inputs and bundle files; return hashed raw timing pairs."""
    matrix = record["cell"]["matrix"]
    _read_hashed_file(repo_root, matrix["path"], matrix["sha256"], "/cell/matrix/path")
    for name in ("identity", "correctness", "measurement"):
        method = record["methodology"][name]
        _read_hashed_file(repo_root, method["path"], method["sha256"],
                          f"/methodology/{name}/path")
    physical_bundle = None
    if bundle_root is not None:
        try:
            physical_bundle = bundle_root.resolve(strict=True)
            _require(physical_bundle.is_dir(), "artifact_path_or_hash", "/bundle_root",
                     "bundle root is not a directory")
        except (OSError, RuntimeError) as exc:
            raise EvidenceError("artifact_path_or_hash", "/bundle_root", str(exc)) from exc
    elif record["artifacts"]:
        raise EvidenceError("artifact_path_or_hash", "/bundle_root",
                            "generated artifacts require a declared bundle root")
    raw_path = record["timing"]["raw_samples_artifact"]
    raw_sides = {name: record["correctness"][name]["raw_artifact"]
                 for name in ("project", "baseline")
                 if record["correctness"][name]["state"] in ("pass", "fail") and
                 record["correctness"][name]["raw_artifact"] is not None}
    if len(raw_sides) == 2:
        _require(raw_sides["project"] != raw_sides["baseline"], "missing_provenance",
                 "/correctness/baseline/raw_artifact", "solver sides need distinct raw outputs")
    seen = set()
    captured = {}
    listed = {artifact["path"]: artifact for artifact in record["artifacts"]}
    for index, artifact in enumerate(record["artifacts"]):
        path = artifact["path"]
        pointer = f"/artifacts/{index}/path"
        _require(path not in seen, "artifact_path_or_hash", pointer,
                 "duplicate bundle artifact path")
        seen.add(path)
        content = _read_hashed_file(physical_bundle, path, artifact["sha256"], pointer,
                                    capture=path in raw_sides.values() or
                                    (record["timing"]["state"] == "measured" and path == raw_path))
        if content is not None:
            captured[path] = content
    for name, path in raw_sides.items():
        if path not in listed:
            continue
        pointer = f"/correctness/{name}/raw_artifact"
        _require(listed[path]["role"] == f"{name}_raw" and
                 listed[path]["media_type"] == "application/json", "missing_provenance",
                 pointer, "solver raw output needs the matching JSON artifact role")
        raw = _parse_json_bytes(captured[path], pointer)
        _require(type(raw) is dict and set(raw) == RAW_SIDE_KEYS, "missing_provenance",
                 pointer, "solver raw output needs status, diagnostics and solution")
        for key in RAW_SIDE_KEYS:
            _require(_strict_equal(raw[key], record["correctness"][name].get(key)),
                     "status_contradiction", f"{pointer}/{key}",
                     "reported solver value differs from raw output")
    review_path = record.get("publication_review_artifact")
    if review_path is not None:
        _require(review_path in seen, "missing_provenance", "/publication_review_artifact",
                 "publication review must cite a listed bundle artifact")
    if record["timing"]["state"] == "measured":
        _require(raw_path in captured, "measured_requires_two_passes_and_raw_samples",
                 "/timing/raw_samples_artifact", "measured record needs hashed raw samples")
        _require(listed[raw_path]["role"] == "timing_raw" and
                 listed[raw_path]["media_type"] == "application/json", "missing_provenance",
                 "/timing/raw_samples_artifact", "timing raw output needs the JSON artifact role")
        return _parse_json_bytes(captured[raw_path], "/timing/raw_samples_artifact")
    return None


def validate_record(record: dict, schema: dict, repo_root: Path,
                    bundle_root: Path | None = None) -> None:
    """Validate a record against its schema and actual referenced files."""
    validate_shape(record, schema)
    raw_samples = verify_artifacts(record, repo_root, bundle_root)
    validate_semantics(record, schema, raw_samples)


def run_self_test(schema: dict, repo_root: Path) -> None:
    """Prove the frozen planning input is valid and selected corruptions fail."""
    fixture = repo_root / "tests/corpus/fixtures/external_baseline_planned_v1.json"
    planning = read_json(fixture)
    validate_record(planning, schema, repo_root)
    cases = (
        ("unknown_field", lambda item: item.__setitem__("unsupported", True),
         "duplicate_or_unknown_field"),
        ("selection", lambda item: item["cell"]["rhs"].__setitem__("values", [1, 2, 3, 4]),
         "selection_mismatch"),
        ("false_pass", lambda item: item["correctness"]["project"].__setitem__("state", "pass"),
         "missing_provenance"),
        ("method_hash", lambda item: item["methodology"]["identity"].__setitem__("sha256", "0" * 64),
         "artifact_path_or_hash"),
    )
    for name, mutate, expected_code in cases:
        candidate = copy.deepcopy(planning)
        mutate(candidate)
        try:
            validate_record(candidate, schema, repo_root)
        except EvidenceError as exc:
            if exc.code == expected_code:
                continue
            raise EvidenceError("implementation_pending", f"/self_test/{name}",
                                f"expected {expected_code}, received {exc.code}") from exc
        raise EvidenceError("implementation_pending", f"/self_test/{name}",
                            "negative fixture was accepted")
    try:
        _parse_json_bytes(b'{"key":1,"key":2}', "/self_test/duplicate_key")
    except EvidenceError as exc:
        if exc.code != "duplicate_or_unknown_field":
            raise EvidenceError("implementation_pending", "/self_test/duplicate_key",
                                f"expected duplicate_or_unknown_field, received {exc.code}") from exc
    else:
        raise EvidenceError("implementation_pending", "/self_test/duplicate_key",
                            "duplicate JSON key was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test == (args.input is not None) or (args.self_test and args.bundle_root):
        parser.error("use --self-test or --input [--bundle-root]")
    record_id = "<unavailable>"
    try:
        schema = read_json(args.schema)
        if schema.get("$id") != SCHEMA_VERSION or schema.get("type") != "object":
            raise EvidenceError("unsupported_enum", "/$id", "unsupported evidence schema")
        repo_root = Path(__file__).resolve().parents[1]
        run_self_test(schema, repo_root)
        if args.self_test:
            print("external-baseline-evidence-v1: self-test PASS")
            return 0
        if args.input is not None:
            record = read_json(args.input)
            if type(record.get("record_id")) is str:
                record_id = record["record_id"]
            validate_record(record, schema, repo_root, args.bundle_root)
            print(f"external-baseline-evidence-v1: valid record_id={record_id} "
                  f"evidence_status={record['evidence_status']}")
            return 0
    except EvidenceError as exc:
        print(f"{exc.code} {exc.pointer} record_id={record_id}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
