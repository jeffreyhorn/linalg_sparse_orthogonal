#!/usr/bin/env python3
"""Validate one Sprint 225 timing cell independently of canonical freshness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "benchmarks/baselines/refactor-csc-nos4"
POLICY = "s225-v2"
TARGET = "SRT-BENCH-REFACTOR-CSC-NOS4"
CLAIM_SCOPE = "one_canonical_refactor_csc_nos4_cell_on_matched_macos_runner"
NON_CLAIMS = [
    "no_portable_speed", "no_cross_platform_timing_parity",
    "no_external_library_ranking", "no_state_of_the_art_claim",
]
FIXTURE_SHA256 = "6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80"
ARGV = ["build/bench_refactor_csc", "tests/data/suitesparse/nos4.mtx", "--repeat", "100"]
RUNNER_FIELDS = (
    "runner_image", "runner_image_version", "os", "architecture", "cpu_model",
    "compiler_path", "compiler_version",
)
IDENTITY_FIELDS = (
    "policy_version", "fixture_sha256", "benchmark_source_sha256", "makefile_sha256",
    "build_recipe", "effective_build", "build_mode", "index_width_bits", "thread_environment",
    "backend_environment", "argv", "timeout_seconds", "warmup_count",
    "measured_count", "metric", "metric_units",
)
TRIAL_FIELDS = (
    "ordinal", "phase", "argv", "started_at_utc", "ended_at_utc",
    "outer_elapsed_ns", "exit_code", "timed_out", "refactor_csc_ms",
    "validity", "stdout_sha256", "stderr_sha256",
)
CSV_FIELDS = (
    "benchmark", "category", "matrix", "scenario", "n", "nnz",
    "ldlt_dense_backend_request", "ldlt_dense_backend_selected",
    "ldlt_dense_backend_fallback", "analyze_ms", "refactor_public_ms",
    "refactor_csc_ms", "solve_public_ms", "solve_csc_ms", "speedup_refactor",
    "res_public", "res_csc",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
MAX_FILE = 2_000_000
MAX_BUNDLE = 20_000_000


class EvidenceError(Exception):
    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"{rule}: {detail}")


def reject(rule: str, detail: str) -> None:
    raise EvidenceError(rule, detail)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_json(data: bytes, rule: str, label: str) -> dict:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        reject(rule, f"{label}: invalid JSON: {exc}")
    if not isinstance(value, dict):
        reject(rule, f"{label}: expected object")
    return value


def file_json(path: Path, rule: str) -> dict:
    try:
        return object_json(path.read_bytes(), rule, str(path))
    except OSError as exc:
        reject(rule, f"{path}: {exc}")


def required_str(value: object, rule: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
        "unknown", "n/a", "not_applicable", "none",
    }:
        reject(rule, f"{label}: missing or sentinel value")
    return value


def sha_field(value: object, rule: str, label: str, pattern: re.Pattern[str] = HEX64) -> str:
    text = required_str(value, rule, label)
    if not pattern.fullmatch(text):
        reject(rule, f"{label}: invalid digest")
    return text


def parse_time(value: object, rule: str, label: str) -> datetime:
    text = required_str(value, rule, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        reject(rule, f"{label}: invalid timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        reject(rule, f"{label}: timestamp must be UTC")
    return parsed


def decimal_value(value: object, rule: str, label: str, positive: bool = True) -> Decimal:
    text = required_str(value, rule, label)
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        reject(rule, f"{label}: invalid decimal")
    if not parsed.is_finite() or (positive and parsed <= 0):
        reject(rule, f"{label}: nonfinite or nonpositive decimal")
    return parsed


def safe_name(name: str) -> bool:
    parts = name.split("/")
    return bool(name) and not name.startswith("/") and "\\" not in name and all(
        part not in ("", ".", "..") for part in parts
    )


def load_bundle(path: Path, missing_rule: str, invalid_rule: str) -> tuple[dict[str, bytes], str]:
    if not path.exists():
        reject(missing_rule, f"bundle missing: {path}")
    members: dict[str, bytes] = {}
    if path.is_dir():
        try:
            files = list(path.rglob("*"))
            for file in files:
                if file.is_symlink():
                    reject(invalid_rule, f"unsafe bundle entry: {file}")
                if file.is_dir():
                    continue
                if not file.is_file():
                    reject(invalid_rule, f"unsafe bundle entry: {file}")
                name = file.relative_to(path).as_posix()
                if not safe_name(name) or file.stat().st_size > MAX_FILE:
                    reject(invalid_rule, f"unsafe or oversized bundle member: {name}")
                members[name] = file.read_bytes()
        except OSError as exc:
            reject(invalid_rule, f"bundle read failed: {exc}")
        bundle_hash = digest(members.get("hashes.sha256", b""))
    elif path.is_file():
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) != len({info.filename for info in infos}):
                    reject(invalid_rule, "duplicate ZIP members")
                if sum(info.file_size for info in infos) > MAX_BUNDLE:
                    reject(invalid_rule, "oversized ZIP bundle")
                for info in infos:
                    name = info.filename
                    if info.is_dir() or not safe_name(name) or info.file_size > MAX_FILE:
                        reject(invalid_rule, f"unsafe or oversized ZIP member: {name}")
                    members[name] = archive.read(info)
            bundle_hash = digest(path.read_bytes())
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            reject(invalid_rule, f"bundle ZIP read failed: {exc}")
    else:
        reject(invalid_rule, f"bundle is not a file or directory: {path}")
    expected = {"run.json", "decision.json", "trials.tsv", "hashes.sha256"}
    expected.update(f"trials/{i:04d}.{stream}" for i in range(1, 26) for stream in ("stdout", "stderr"))
    missing = expected - set(members)
    if missing:
        reject(missing_rule, f"bundle members missing: {sorted(missing)}")
    if set(members) != expected:
        reject(invalid_rule, f"bundle has extra members: {sorted(set(members)-expected)}")
    try:
        lines = members["hashes.sha256"].decode("ascii").splitlines()
    except UnicodeDecodeError:
        reject(invalid_rule, "non-ASCII hash manifest")
    hashes: dict[str, str] = {}
    for line in lines:
        if "  " not in line:
            reject(invalid_rule, "malformed hash manifest row")
        value, name = line.split("  ", 1)
        if not HEX64.fullmatch(value) or not safe_name(name) or name in hashes:
            reject(invalid_rule, "invalid or duplicate hash manifest entry")
        hashes[name] = value
    if set(hashes) != expected - {"hashes.sha256"}:
        reject(invalid_rule, "hash manifest membership mismatch")
    for name, value in hashes.items():
        if digest(members[name]) != value:
            reject(invalid_rule, f"hash mismatch: {name}")
    return members, bundle_hash


def check_run(members: dict[str, bytes], role: str, rule: str) -> dict:
    run = object_json(members["run.json"], rule, "run.json")
    receipt = object_json(members["decision.json"], rule, "decision.json")
    if run.get("schema_version") != 2 or run.get("policy_version") != POLICY or run.get("role") != role:
        reject(rule, "run schema/policy/role mismatch")
    if run.get("dry_run") is not False or run.get("source_dirty") is not False:
        reject(rule, "dry run or dirty checkout")
    sha_field(run.get("source_commit"), rule, "source_commit", HEX40)
    for key in ("executable_sha256", "library_sha256", "fixture_sha256", "benchmark_source_sha256", "makefile_sha256"):
        sha_field(run.get(key), rule, key)
    if run["fixture_sha256"] != FIXTURE_SHA256:
        reject(rule, "fixture digest mismatch")
    for key in ("workflow_run_id", "workflow_run_attempt"):
        value = required_str(run.get(key), rule, key)
        if not value.isdecimal() or int(value) < 1:
            reject(rule, f"{key}: invalid hosted identity")
    required_str(run.get("workflow_job_key"), rule, "workflow_job_key")
    runner = run.get("runner")
    if not isinstance(runner, dict):
        reject(rule, "runner metadata missing")
    for key in RUNNER_FIELDS:
        required_str(runner.get(key), rule, f"runner.{key}")
    if not runner["runner_image"].lower().startswith("macos") or not runner["os"].lower().startswith("macos"):
        reject(rule, "runner is outside the locked macOS class")
    if run.get("argv") != ARGV or run.get("build_recipe") != "make build/bench_refactor_csc SPARSE_IDX_BITS=32":
        reject(rule, "command or build recipe mismatch")
    if run.get("build_mode") != "serial" or run.get("index_width_bits") != 32:
        reject(rule, "build mode or index width mismatch")
    if run.get("thread_environment") != {"OMP_NUM_THREADS": "1", "SPARSE_OPENMP": "unset"}:
        reject(rule, "thread environment mismatch")
    if run.get("backend_environment") != {
        "SPARSE_LDLT_DENSE_BACKEND": "unset", "SPARSE_CHOL_DENSE_BACKEND": "unset",
        "SPARSE_SUPERNODAL_POSTORDER": "unset", "SPARSE_ND_SUPERNODAL_POSTORDER": "unset",
    }:
        reject(rule, "backend environment mismatch")
    build = run.get("effective_build")
    if not isinstance(build, dict) or set(build) != {
        "cc", "sysroot", "cflags", "ldflags", "sparse_cflags",
    }:
        reject(rule, "effective build settings missing")
    sysroot = build["sysroot"]
    if not isinstance(sysroot, str) or not sysroot.startswith("/") or "\n" in sysroot:
        reject(rule, "invalid SDK sysroot")
    expected_cflags = "-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2"
    if sysroot:
        expected_cflags += f" -isysroot {sysroot}"
    if (build["cc"], build["cflags"], build["ldflags"], build["sparse_cflags"]) != (
        "cc", expected_cflags, "-lm", expected_cflags + " -DSPARSE_IDX_BITS=32"
    ):
        reject(rule, "effective compiler, flags, or feature settings differ from locked build")
    if (run.get("timeout_seconds"), run.get("warmup_count"), run.get("measured_count")) != (30, 5, 20):
        reject(rule, "trial schedule mismatch")
    if run.get("metric") != "refactor_csc_ms" or run.get("metric_units") != "ms":
        reject(rule, "metric mismatch")
    run_start = parse_time(run.get("started_at_utc"), rule, "started_at_utc")
    run_end = parse_time(run.get("ended_at_utc"), rule, "ended_at_utc")
    if not timedelta(0) <= run_end - run_start <= timedelta(minutes=20):
        reject(rule, "run timestamps reversed or 20-minute cap exceeded")
    expected_status = "pending_baseline_review" if role == "baseline" else "pending_timing_review"
    if (receipt.get("policy_version"), receipt.get("role"), receipt.get("status"),
        receipt.get("dry_run"),
        receipt.get("baseline_accepted"), receipt.get("freshness_result")) != (
        POLICY, role, expected_status, False, False, "separate"
    ):
        reject(rule, "collector receipt mismatch")
    if receipt.get("trial_count") != 25:
        reject("S225_CANDIDATE_INVALID" if role == "candidate" else rule,
               "collector receipt does not contain 25 trials")
    if type(receipt.get("all_trials_valid")) is not bool or (role == "baseline" and not receipt["all_trials_valid"]):
        reject(rule, "collector validity receipt mismatch")
    return run


def parse_metric(stdout: bytes, rule: str, ordinal: int) -> Decimal:
    try:
        rows = list(csv.reader(io.StringIO(stdout.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error):
        reject(rule, f"trial {ordinal}: invalid raw CSV")
    if len(rows) != 2 or tuple(rows[0]) != CSV_FIELDS or len(rows[1]) != len(CSV_FIELDS):
        reject(rule, f"trial {ordinal}: wrong CSV shape")
    row = dict(zip(CSV_FIELDS, rows[1]))
    if tuple(row[k] for k in ("benchmark", "category", "matrix", "scenario", "n", "nnz")) != (
        "bench_refactor_csc", "proof", "nos4.mtx", "chol_spd", "100", "594"
    ):
        reject(rule, f"trial {ordinal}: wrong selected row")
    metric = row["refactor_csc_ms"]
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{3}", metric):
        reject(rule, f"trial {ordinal}: metric does not have locked millisecond precision")
    value = decimal_value(metric, rule, f"trial {ordinal} raw metric")
    if value > Decimal("30000"):
        reject(rule, f"trial {ordinal}: metric exceeds the 30-second process cap")
    return value


def check_trials(members: dict[str, bytes], run: dict, rule: str) -> list[Decimal]:
    try:
        reader = csv.DictReader(io.StringIO(members["trials.tsv"].decode("utf-8")), delimiter="\t")
        if tuple(reader.fieldnames or ()) != TRIAL_FIELDS:
            reject(rule, "trials.tsv header mismatch")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error):
        reject(rule, "trials.tsv is not valid UTF-8 TSV")
    if len(rows) != 25:
        reject(rule, f"expected 25 trials, got {len(rows)}")
    measured: list[Decimal] = []
    for ordinal, row in enumerate(rows, start=1):
        if None in row or any(value is None for value in row.values()):
            reject(rule, f"trial {ordinal}: malformed TSV row")
        if row["ordinal"] != str(ordinal) or row["phase"] != ("warmup" if ordinal <= 5 else "measured"):
            reject(rule, f"trial {ordinal}: ordinal or phase mismatch")
        try:
            argv = json.loads(row["argv"])
        except json.JSONDecodeError:
            reject(rule, f"trial {ordinal}: invalid argv")
        if argv != ARGV or argv != run["argv"]:
            reject(rule, f"trial {ordinal}: argv mismatch")
        if row["exit_code"] != "0" or row["timed_out"] != "False" or row["validity"] != "valid":
            reject(rule, f"trial {ordinal}: failed, timed out, or invalid")
        trial_start = parse_time(row["started_at_utc"], rule, f"trial {ordinal} start")
        trial_end = parse_time(row["ended_at_utc"], rule, f"trial {ordinal} end")
        if not (parse_time(run["started_at_utc"], rule, "run start") <= trial_start <= trial_end <=
                parse_time(run["ended_at_utc"], rule, "run end")):
            reject(rule, f"trial {ordinal}: timestamps outside run interval")
        if not row["outer_elapsed_ns"].isdecimal() or not 0 < int(row["outer_elapsed_ns"]) <= 30_000_000_000:
            reject(rule, f"trial {ordinal}: invalid elapsed time")
        for stream in ("stdout", "stderr"):
            path = f"trials/{ordinal:04d}.{stream}"
            if sha_field(row[f"{stream}_sha256"], rule, f"trial {ordinal} {stream} digest") != digest(members[path]):
                reject(rule, f"trial {ordinal}: {stream} digest mismatch")
        raw = parse_metric(members[f"trials/{ordinal:04d}.stdout"], rule, ordinal)
        if decimal_value(row["refactor_csc_ms"], rule, f"trial {ordinal} TSV metric") != raw:
            reject(rule, f"trial {ordinal}: TSV/raw metric mismatch")
        if ordinal > 5:
            measured.append(raw)
    return measured


def stats(values: list[Decimal]) -> dict[str, Decimal]:
    mid = median(values)
    mad = median([abs(value - mid) for value in values])
    p90 = sorted(values)[17]
    return {"median_ms": mid, "mad_ms": mad, "mad_ratio": mad / mid,
            "p90_ms": p90, "p90_ratio": p90 / mid}


def fingerprint(run: dict) -> dict:
    return {
        **{key: run[key] for key in IDENTITY_FIELDS},
        "runner": {key: run["runner"][key] for key in RUNNER_FIELDS},
    }


def compare_stat(actual: Decimal, recorded: object, rule: str, label: str) -> None:
    if actual != decimal_value(recorded, rule, label, positive=False):
        reject(rule, f"{label}: recorded statistic disagrees with raw data")


def verify_candidate_checkout(run: dict, checkout_root: Path) -> None:
    """Bind a candidate receipt to the files available to the hosted decider."""
    rule = "S225_CANDIDATE_IDENTITY"
    files = {
        "fixture_sha256": "tests/data/suitesparse/nos4.mtx",
        "benchmark_source_sha256": "benchmarks/bench_refactor_csc.c",
        "makefile_sha256": "Makefile",
        "executable_sha256": "build/bench_refactor_csc",
        "library_sha256": "build/libsparse_lu_ortho.a",
    }
    for field, name in files.items():
        path = checkout_root / name
        try:
            actual = digest(path.read_bytes())
        except OSError as exc:
            reject(rule, f"candidate checkout {name}: {exc}")
        if actual != run[field]:
            reject(rule, f"candidate checkout {name}: digest differs from raw collection")
    if run["fixture_sha256"] != FIXTURE_SHA256:
        reject(rule, "candidate fixture differs from locked policy")
    stamp = checkout_root / "build/.sparse-idx-bits"
    try:
        width = stamp.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        reject(rule, f"candidate width stamp: {exc}")
    if width != "32":
        reject(rule, "candidate build width stamp is not 32")


def load_baseline(root: Path, now: datetime, require_current: bool = True) -> dict:
    missing = "S225_BASELINE_MISSING"
    invalid = "S225_BASELINE_INVALID"
    pointer_path = root / "active.json"
    if not pointer_path.is_file():
        reject(missing, f"accepted baseline pointer missing: {pointer_path}")
    pointer = file_json(pointer_path, invalid)
    baseline_id = required_str(pointer.get("baseline_id"), invalid, "baseline_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", baseline_id):
        reject(invalid, "invalid baseline_id")
    if pointer.get("status") != "accepted" or pointer.get("policy_version") != POLICY:
        reject(invalid, f"active pointer is not accepted {POLICY}")
    manifest_path = root / POLICY / baseline_id / "baseline.json"
    if not manifest_path.is_file():
        reject(missing, f"baseline manifest missing: {manifest_path}")
    raw_manifest = manifest_path.read_bytes()
    manifest_hash = digest(raw_manifest)
    if sha_field(pointer.get("baseline_manifest_sha256"), invalid, "baseline manifest digest") != manifest_hash:
        reject(invalid, "baseline manifest digest mismatch")
    manifest = object_json(raw_manifest, invalid, "baseline.json")
    if (manifest.get("schema_version"), manifest.get("policy_version"), manifest.get("status"),
        manifest.get("target_id"), manifest.get("baseline_id")) != (1, POLICY, "accepted", TARGET, baseline_id):
        reject(invalid, "baseline manifest identity or approval mismatch")
    required_str(manifest.get("approved_by"), invalid, "approved_by")
    source = sha_field(manifest.get("source_commit"), invalid, "baseline source commit", HEX40)
    accepted = parse_time(manifest.get("accepted_at_utc"), invalid, "accepted_at_utc")
    expires = parse_time(manifest.get("expires_at_utc"), invalid, "expires_at_utc")
    if accepted > now or (require_current and expires <= now):
        reject(invalid, "baseline not yet accepted or expired")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 3:
        reject(invalid, "baseline must contain exactly three jobs")
    baseline_dir = manifest_path.parent
    seen_ids: set[tuple[str, str]] = set()
    reference_fingerprint: dict | None = None
    reference_executable: str | None = None
    medians: list[Decimal] = []
    bundle_hashes: list[str] = []
    latest_end = datetime.min.replace(tzinfo=timezone.utc)
    earliest_end = datetime.max.replace(tzinfo=timezone.utc)
    for number, job in enumerate(jobs, 1):
        if not isinstance(job, dict):
            reject(invalid, f"job {number}: expected object")
        name = required_str(job.get("bundle_file"), invalid, f"job {number} bundle_file")
        if Path(name).name != name or not name.endswith(".zip"):
            reject(invalid, f"job {number}: unsafe bundle filename")
        bundle_path = baseline_dir / name
        if not bundle_path.is_file():
            reject(missing, f"baseline raw bundle missing: {bundle_path}")
        members, bundle_hash = load_bundle(bundle_path, missing, invalid)
        if sha_field(job.get("bundle_sha256"), invalid, f"job {number} digest") != bundle_hash:
            reject(invalid, f"job {number}: ZIP digest mismatch")
        run = check_run(members, "baseline", invalid)
        if run["source_commit"] != source:
            reject(invalid, f"job {number}: source commit mismatch")
        identity = (run["workflow_run_id"], run["workflow_run_attempt"])
        if identity in seen_ids:
            reject(invalid, f"job {number}: duplicate hosted run attempt")
        seen_ids.add(identity)
        if job.get("run_id") != identity[0] or str(job.get("attempt")) != identity[1] or job.get("job_key") != run["workflow_job_key"]:
            reject(invalid, f"job {number}: hosted run identity mismatch")
        current_fingerprint = fingerprint(run)
        if reference_fingerprint is None:
            reference_fingerprint = current_fingerprint
            reference_executable = run["executable_sha256"]
        elif current_fingerprint != reference_fingerprint or run["executable_sha256"] != reference_executable:
            reject(invalid, f"job {number}: runner/build/executable fingerprint mismatch")
        job_end = parse_time(run["ended_at_utc"], invalid, f"job {number} end")
        latest_end = max(latest_end, job_end)
        earliest_end = min(earliest_end, job_end)
        values = check_trials(members, run, invalid)
        result = stats(values)
        if result["mad_ratio"] > Decimal("0.10") or result["p90_ratio"] > Decimal("1.50"):
            reject(invalid, f"job {number}: baseline dispersion exceeded")
        for key in ("median_ms", "mad_ratio", "p90_ratio"):
            compare_stat(result[key], job.get(key), invalid, f"job {number} {key}")
        medians.append(result["median_ms"])
        bundle_hashes.append(bundle_hash)
    if accepted < latest_end or expires > earliest_end + timedelta(days=30) or expires <= accepted:
        reject(invalid, "baseline acceptance or expiry timestamps invalid")
    spread = (max(medians) - min(medians)) / median(medians)
    if spread > Decimal("0.10"):
        reject(invalid, "baseline between-job dispersion exceeded")
    compare_stat(spread, manifest.get("between_job_spread"), invalid, "between_job_spread")
    reference = median(medians)
    compare_stat(reference, manifest.get("reference_median_ms"), invalid, "reference_median_ms")
    if manifest.get("fingerprint") != reference_fingerprint:
        reject(invalid, "baseline manifest fingerprint mismatch")
    return {"id": baseline_id, "manifest_sha256": manifest_hash, "bundle_hashes": bundle_hashes,
            "fingerprint": reference_fingerprint, "reference_ms": reference, "spread": spread,
            "medians": medians, "accepted_at_utc": manifest["accepted_at_utc"],
            "expires_at_utc": manifest["expires_at_utc"]}


def decision(baseline_root: Path, candidate_path: Path | None,
             expected_commit: str | None = None,
             checkout_root: Path | None = None,
             evaluation_time: datetime | None = None) -> tuple[dict, int]:
    evaluated = evaluation_time or datetime.now(timezone.utc)
    report = {
        "schema_version": 2, "policy_version": POLICY, "target_id": TARGET,
        "claim_scope": CLAIM_SCOPE, "non_claims": NON_CLAIMS.copy(),
        "status": "blocked", "primary_rule": None, "secondary_rules": [], "detail": None,
        "freshness_result": "separate", "baseline_id": None,
        "baseline_manifest_sha256": None, "baseline_bundle_sha256": None,
        "baseline_job_medians_ms": None, "baseline_between_job_spread": None,
        "candidate_bundle_sha256": None, "candidate_hashes_sha256": None,
        "candidate_source_commit": None, "candidate_run_id": None,
        "candidate_attempt": None, "candidate_job_key": None,
        "runner_fingerprint": None, "warmup_count": None, "measured_count": None,
        "invalid_count": None, "candidate_median_ms": None, "candidate_mad_ms": None,
        "candidate_mad_ratio": None, "candidate_p90_ms": None,
        "candidate_p90_ratio": None, "reference_median_ms": None,
        "allowed_ceiling_ms": None, "baseline_accepted_at_utc": None,
        "baseline_expires_at_utc": None, "candidate_started_at_utc": None,
        "candidate_ended_at_utc": None,
        "evaluated_at_utc": evaluated.isoformat(),
    }
    try:
        if evaluated.tzinfo is None or evaluated.utcoffset() != timedelta(0) or evaluated > datetime.now(timezone.utc):
            reject("S225_CANDIDATE_IDENTITY", "evaluation time is not a past UTC instant")
        baseline = load_baseline(baseline_root, evaluated, require_current=candidate_path is None)
        report.update({"baseline_id": baseline["id"],
                       "baseline_manifest_sha256": baseline["manifest_sha256"],
                       "baseline_bundle_sha256": baseline["bundle_hashes"],
                       "baseline_job_medians_ms": [str(x) for x in baseline["medians"]],
                       "baseline_between_job_spread": str(baseline["spread"]),
                       "reference_median_ms": str(baseline["reference_ms"]),
                       "baseline_accepted_at_utc": baseline["accepted_at_utc"],
                       "baseline_expires_at_utc": baseline["expires_at_utc"]})
        if candidate_path is None:
            report.update({"status": "ready", "primary_rule": "S225_BASELINE_READY",
                           "detail": "accepted baseline verified; no candidate decision made"})
            return report, 0
        members, candidate_hash = load_bundle(candidate_path, "S225_CANDIDATE_MISSING", "S225_CANDIDATE_INVALID")
        report["candidate_bundle_sha256"] = candidate_hash
        report["candidate_hashes_sha256"] = digest(members["hashes.sha256"])
        run = check_run(members, "candidate", "S225_CANDIDATE_IDENTITY")
        started = parse_time(run["started_at_utc"], "S225_CANDIDATE_IDENTITY", "candidate start")
        ended = parse_time(run["ended_at_utc"], "S225_CANDIDATE_IDENTITY", "candidate end")
        accepted = parse_time(baseline["accepted_at_utc"], "S225_BASELINE_INVALID", "accepted_at_utc")
        expires = parse_time(baseline["expires_at_utc"], "S225_BASELINE_INVALID", "expires_at_utc")
        if not accepted <= started <= ended < expires or ended > evaluated:
            reject("S225_CANDIDATE_IDENTITY", "candidate collection outside baseline validity or evaluation time")
        report.update({"candidate_source_commit": run["source_commit"],
                       "candidate_run_id": run["workflow_run_id"],
                       "candidate_attempt": run["workflow_run_attempt"],
                       "candidate_job_key": run["workflow_job_key"],
                       "candidate_started_at_utc": run["started_at_utc"],
                       "candidate_ended_at_utc": run["ended_at_utc"],
                       "runner_fingerprint": fingerprint(run),
                       "warmup_count": 5, "measured_count": 20})
        if expected_commit is not None and run["source_commit"] != expected_commit:
            reject("S225_CANDIDATE_IDENTITY", "candidate commit differs from checkout")
        if checkout_root is not None:
            verify_candidate_checkout(run, checkout_root)
        if fingerprint(run) != baseline["fingerprint"]:
            reject("S225_CANDIDATE_IDENTITY", "candidate runner/workload/build fingerprint differs from baseline")
        values = check_trials(members, run, "S225_CANDIDATE_INVALID")
        if object_json(members["decision.json"], "S225_CANDIDATE_INVALID", "decision.json").get("all_trials_valid") is not True:
            reject("S225_CANDIDATE_INVALID", "collector receipt disagrees with valid raw trials")
        report["invalid_count"] = 0
        result = stats(values)
        report.update({"candidate_median_ms": str(result["median_ms"]),
                       "candidate_mad_ms": str(result["mad_ms"]),
                       "candidate_mad_ratio": str(result["mad_ratio"]),
                       "candidate_p90_ms": str(result["p90_ms"]),
                       "candidate_p90_ratio": str(result["p90_ratio"]),
                       "allowed_ceiling_ms": str(baseline["reference_ms"] * Decimal("1.25"))})
        if result["mad_ratio"] > Decimal("0.10") or result["p90_ratio"] > Decimal("1.50"):
            reject("S225_CANDIDATE_DISPERSION", "candidate dispersion exceeds locked limits")
        if result["median_ms"] > baseline["reference_ms"] * Decimal("1.25"):
            reject("S225_TIMING_REGRESSION", "candidate median exceeds locked ceiling")
        report.update({"status": "pass", "primary_rule": "S225_TIMING_PASS", "detail": "one-cell timing gate passed"})
        return report, 0
    except EvidenceError as exc:
        report["primary_rule"] = exc.rule
        report["detail"] = exc.detail
        if exc.rule == "S225_CANDIDATE_DISPERSION":
            report["status"] = "unstable"
        elif exc.rule == "S225_TIMING_REGRESSION":
            report["status"] = "regression"
        elif exc.rule.endswith("INVALID") or exc.rule == "S225_CANDIDATE_IDENTITY":
            report["status"] = "invalid"
        else:
            report["status"] = "blocked"
        return report, 1 if exc.rule in ("S225_CANDIDATE_DISPERSION", "S225_TIMING_REGRESSION") else 2


def verify_report_against_evidence(report: dict, baseline_root: Path,
                                   candidate_path: Path | None,
                                   expected_commit: str | None = None,
                                   checkout_root: Path | None = None) -> int:
    """Reject a decision or claim that cannot be reproduced from raw evidence."""
    try:
        evaluated = parse_time(report.get("evaluated_at_utc"), "S225_REPORT_INVALID", "evaluated_at_utc")
    except (AttributeError, TypeError):
        reject("S225_REPORT_INVALID", "report is not an object")
    if evaluated > datetime.now(timezone.utc):
        reject("S225_REPORT_INVALID", "report evaluation time is in the future")
    expected, code = decision(baseline_root, candidate_path, expected_commit, checkout_root,
                              evaluation_time=evaluated)
    if report != expected:
        reject("S225_REPORT_INVALID", "report differs from recomputed raw-evidence decision")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--preflight", action="store_true", help="verify baseline only; never issue a timing pass")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-candidate-commit")
    parser.add_argument("--verify-checkout", action="store_true",
                        help="compare candidate receipt hashes with this checkout and build")
    args = parser.parse_args()
    if args.preflight == (args.candidate is not None):
        parser.error("select exactly one of --preflight or --candidate")
    if args.expected_candidate_commit and not HEX40.fullmatch(args.expected_candidate_commit):
        parser.error("--expected-candidate-commit requires a full lowercase SHA-1")
    if args.verify_checkout and (args.preflight or not args.expected_candidate_commit):
        parser.error("--verify-checkout requires --candidate and --expected-candidate-commit")
    report, exit_code = decision(args.baseline_root, args.candidate, args.expected_candidate_commit,
                                 ROOT if args.verify_checkout else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"refactor-csc-timing: {report['primary_rule']}: {report['detail']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
