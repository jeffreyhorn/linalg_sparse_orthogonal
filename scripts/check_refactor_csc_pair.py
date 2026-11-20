#!/usr/bin/env python3
"""Recompute the Sprint 225 same-machine timing decision from three raw pairs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from statistics import median

import check_refactor_csc_timing as common
from collect_refactor_csc_pair import (
    BASELINE_COMMIT, COMMAND, FIELDS, MEASURED_PAIRS, POLICY,
    TIMEOUT_SECONDS, WARMUP_PAIRS,
)


ROOT = Path(__file__).resolve().parents[1]
RULE = "S225_PAIR_INVALID"
EXPECTED_ARGV = ["build/bench_refactor_csc", *COMMAND]
RUNNER_CLASS = (
    "runner_image", "runner_image_version", "os", "architecture",
    "cpu_model", "compiler_path", "compiler_version",
)
SHA_FIELDS = ("executable_sha256", "library_sha256", "benchmark_source_sha256",
              "makefile_sha256", "fixture_sha256")
REQUIRED = {"run.json", "decision.json", "trials.tsv", "hashes.sha256",
            "build-baseline.log", "build-candidate.log"}
REQUIRED.update(
    f"trials/{pair:04d}-{side}.{stream}"
    for pair in range(1, WARMUP_PAIRS + MEASURED_PAIRS + 1)
    for side in ("baseline", "candidate") for stream in ("stdout", "stderr")
)


def reject(detail: str, rule: str = RULE) -> None:
    raise common.EvidenceError(rule, detail)


def read_bundle(path: Path) -> tuple[dict[str, bytes], str]:
    if not path.exists():
        reject(f"missing raw pair: {path}", "S225_PAIR_MISSING")
    members: dict[str, bytes] = {}
    if path.is_dir():
        for file in path.rglob("*"):
            if file.is_symlink() or (not file.is_file() and not file.is_dir()):
                reject(f"unsafe member: {file}")
            if file.is_file():
                name = file.relative_to(path).as_posix()
                if not common.safe_name(name) or file.stat().st_size > common.MAX_FILE:
                    reject(f"unsafe member: {name}")
                members[name] = file.read_bytes()
    elif path.is_file():
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) != len({info.filename for info in infos}):
                    reject("duplicate ZIP member")
                if sum(info.file_size for info in infos) > common.MAX_BUNDLE:
                    reject("oversized ZIP")
                for info in infos:
                    if info.is_dir() or not common.safe_name(info.filename) or info.file_size > common.MAX_FILE:
                        reject(f"unsafe ZIP member: {info.filename}")
                    members[info.filename] = archive.read(info)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            reject(f"cannot read ZIP: {exc}")
    else:
        reject(f"raw pair is not a file or directory: {path}")
    if set(members) != REQUIRED:
        reject(f"pair member set mismatch: missing={sorted(REQUIRED-set(members))}, extra={sorted(set(members)-REQUIRED)}")
    member_manifest_digest = common.digest(members["hashes.sha256"])
    try:
        lines = members["hashes.sha256"].decode("ascii").splitlines()
    except UnicodeDecodeError:
        reject("non-ASCII hash manifest")
    hashes: dict[str, str] = {}
    for line in lines:
        if "  " not in line:
            reject("malformed hash row")
        value, name = line.split("  ", 1)
        if not common.HEX64.fullmatch(value) or not common.safe_name(name) or name in hashes:
            reject("invalid or duplicate hash row")
        hashes[name] = value
    if set(hashes) != REQUIRED - {"hashes.sha256"}:
        reject("hash manifest member set mismatch")
    for name, value in hashes.items():
        if common.digest(members[name]) != value:
            reject(f"hash mismatch: {name}")
    return members, member_manifest_digest


@lru_cache(maxsize=32)
def source_sha(commit: str, name: str, checkout: Path) -> str:
    try:
        value = subprocess.check_output(["git", "-C", str(checkout), "show", f"{commit}:{name}"],
                                        stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        reject(f"cannot verify source file {commit}:{name}")
    return common.digest(value)


def check_build(side: object, commit: str, checkout: Path) -> dict:
    if not isinstance(side, dict) or side.get("source_commit") != commit or side.get("source_dirty") is not False:
        reject("source commit or clean state mismatch")
    for field in SHA_FIELDS:
        common.sha_field(side.get(field), RULE, field)
    for field, path in (
        ("benchmark_source_sha256", "benchmarks/bench_refactor_csc.c"),
        ("makefile_sha256", "Makefile"),
        ("fixture_sha256", "tests/data/suitesparse/nos4.mtx"),
    ):
        if side[field] != source_sha(commit, path, checkout):
            reject(f"{field} differs from committed source")
    if side["fixture_sha256"] != common.FIXTURE_SHA256:
        reject("fixture SHA differs from locked nos4 input")
    build = side.get("effective_build")
    if not isinstance(build, dict) or set(build) != {"cc", "sysroot", "cflags", "ldflags", "sparse_cflags"}:
        reject("effective build receipt missing")
    sysroot = build["sysroot"]
    if not isinstance(sysroot, str) or not sysroot.startswith("/") or "\n" in sysroot:
        reject("invalid SDK sysroot")
    flags = f"-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2 -isysroot {sysroot}"
    if (build["cc"], build["cflags"], build["ldflags"], build["sparse_cflags"]) != (
        "cc", flags, "-lm", flags + " -DSPARSE_IDX_BITS=32",
    ):
        reject("effective build differs from pinned serial 32-bit recipe")
    return side


def stats(values: list[Decimal]) -> dict[str, Decimal]:
    mid = median(values)
    mad = median([abs(value - mid) for value in values])
    p90 = sorted(values)[17]
    return {"median": mid, "mad_ratio": mad / mid, "p90_ratio": p90 / mid}


def check_pair(path: Path, replica: int, candidate_commit: str, checkout: Path,
               expected_run: str, expected_attempt: str) -> tuple[dict, str]:
    members, member_manifest_hash = read_bundle(path)
    run = common.object_json(members["run.json"], RULE, "run.json")
    receipt = common.object_json(members["decision.json"], RULE, "decision.json")
    if (run.get("schema_version"), run.get("policy_version"), run.get("baseline_commit"),
        run.get("candidate_commit"), run.get("replica")) != (
        1, POLICY, BASELINE_COMMIT, candidate_commit, replica,
    ):
        reject("run schema, policy, source commit, or replica mismatch")
    if not isinstance(run.get("workflow_run_id"), str) or not run["workflow_run_id"].isdecimal():
        reject("missing hosted run ID")
    if run["workflow_run_id"] != expected_run:
        reject("run ID differs from hosted decision job")
    if run.get("workflow_run_attempt") != expected_attempt:
        reject("run attempt differs from hosted decision job")
    common.required_str(run.get("workflow_job_key"), RULE, "workflow_job_key")
    runner = run.get("runner")
    if not isinstance(runner, dict):
        reject("missing runner identity")
    for field in RUNNER_CLASS:
        common.required_str(runner.get(field), RULE, f"runner.{field}")
    common.required_str(runner.get("runner_label"), RULE, "runner.runner_label")
    if not (runner["runner_image"].lower().startswith("macos") and
            runner["os"].lower().startswith("macos") and runner["architecture"] == "x86_64"):
        reject("runner is outside the reviewed macOS Intel class")
    if (run.get("argv"), run.get("warmup_pairs"), run.get("measured_pairs"),
        run.get("timeout_seconds_per_side"), run.get("metric"), run.get("metric_units"),
        run.get("thread_environment")) != (
        EXPECTED_ARGV, WARMUP_PAIRS, MEASURED_PAIRS, TIMEOUT_SECONDS,
        "refactor_csc_ms", "ms", {"OMP_NUM_THREADS": "1", "SPARSE_OPENMP": "unset"},
    ):
        reject("paired workload or schedule mismatch")
    baseline = check_build(run.get("baseline"), BASELINE_COMMIT, checkout)
    candidate = check_build(run.get("candidate"), candidate_commit, checkout)
    if baseline["benchmark_source_sha256"] != candidate["benchmark_source_sha256"]:
        reject("baseline and candidate benchmark source differ")
    if baseline["effective_build"] != candidate["effective_build"] or baseline["fixture_sha256"] != candidate["fixture_sha256"]:
        reject("baseline and candidate build or fixture differ")
    if receipt != {
        "policy_version": POLICY, "status": "pending_pair_review", "all_trials_valid": True,
        "trial_count": 50, "baseline_accepted": False, "freshness_result": "separate",
    }:
        reject("collector receipt disagrees with raw paired schedule")
    start = common.parse_time(run.get("started_at_utc"), RULE, "run start")
    end = common.parse_time(run.get("ended_at_utc"), RULE, "run end")
    if not start <= end <= datetime.now(timezone.utc) or end - start > timedelta(minutes=20):
        reject("invalid run timestamps or 20-minute cap")
    try:
        reader = csv.DictReader(io.StringIO(members["trials.tsv"].decode("utf-8")), delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            reject("trials.tsv header mismatch")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error):
        reject("malformed trial table")
    if len(rows) != 50:
        reject("paired trial count is not 50")
    values: list[Decimal] = []
    previous_end = start
    for pair in range(1, WARMUP_PAIRS + MEASURED_PAIRS + 1):
        order = ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")
        pair_values: dict[str, Decimal] = {}
        for offset, side in enumerate(order):
            row = rows[2 * (pair - 1) + offset]
            if None in row or any(value is None for value in row.values()):
                reject(f"pair {pair}: malformed trial row")
            if (row["pair"], row["phase"], row["side"], row["exit_code"], row["timed_out"], row["validity"]) != (
                str(pair), "warmup" if pair <= WARMUP_PAIRS else "measured", side, "0", "False", "valid",
            ):
                reject(f"pair {pair}: order, phase, side, or execution status mismatch")
            try:
                argv = json.loads(row["argv"])
            except json.JSONDecodeError:
                reject(f"pair {pair}: invalid argv")
            if argv != EXPECTED_ARGV:
                reject(f"pair {pair}: argv differs from reviewed command")
            trial_start = common.parse_time(row["started_at_utc"], RULE, "trial start")
            trial_end = common.parse_time(row["ended_at_utc"], RULE, "trial end")
            if not previous_end <= trial_start <= trial_end <= end:
                reject(f"pair {pair}: trial timestamps out of sequence")
            previous_end = trial_end
            if not row["outer_elapsed_ns"].isdecimal() or not 0 < int(row["outer_elapsed_ns"]) <= 30_000_000_000:
                reject(f"pair {pair}: process duration exceeds 30 seconds")
            stem = f"trials/{pair:04d}-{side}"
            for stream in ("stdout", "stderr"):
                if common.sha_field(row[f"{stream}_sha256"], RULE, f"{stem}.{stream} digest") != common.digest(members[f"{stem}.{stream}"]):
                    reject(f"pair {pair}: raw {stream} hash mismatch")
            metric = common.parse_metric(members[f"{stem}.stdout"], RULE, pair)
            if common.decimal_value(row["refactor_csc_ms"], RULE, "TSV metric") != metric:
                reject(f"pair {pair}: raw/TSV metric mismatch")
            pair_values[side] = metric
        if pair > WARMUP_PAIRS:
            values.append(pair_values["candidate"] / pair_values["baseline"])
    result = stats(values)
    return {
        "replica": replica, "run_id": run["workflow_run_id"],
        "attempt": run["workflow_run_attempt"], "job_key": run["workflow_job_key"],
        "runner": {field: runner[field] for field in RUNNER_CLASS},
        "baseline_executable_sha256": baseline["executable_sha256"],
        "candidate_executable_sha256": candidate["executable_sha256"],
        "baseline_build": baseline["effective_build"],
        "candidate_build": candidate["effective_build"],
        "median_ratio": str(result["median"]), "mad_ratio": str(result["mad_ratio"]),
        "p90_ratio": str(result["p90_ratio"]),
    }, member_manifest_hash


def decide(root: Path, candidate_commit: str, checkout: Path,
           expected_run: str, expected_attempt: str) -> tuple[dict, int]:
    report: dict[str, object] = {
        "schema_version": 1, "policy_version": POLICY,
        "baseline_commit": BASELINE_COMMIT, "candidate_commit": candidate_commit,
        "status": "blocked", "primary_rule": "S225_PAIR_MISSING",
        "claim_scope": "one_paired_refactor_csc_nos4_cell_on_matched_macos_runner",
        "non_claims": common.NON_CLAIMS, "jobs": [], "between_job_spread": None,
        "reference_median_ratio": None, "detail": "raw pair evidence missing",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", candidate_commit):
            reject("invalid candidate source commit")
        if not expected_run.isdecimal() or not expected_attempt.isdecimal():
            reject("invalid hosted run or attempt identity")
        if subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() != candidate_commit:
            reject("candidate checkout differs from claimed commit")
        jobs = []
        for replica in (1, 2, 3):
            result, member_manifest_hash = check_pair(root / f"sprint225-pair-{expected_run}-{expected_attempt}-{replica}",
                                             replica, candidate_commit, checkout,
                                             expected_run, expected_attempt)
            result["member_manifest_sha256"] = member_manifest_hash
            jobs.append(result)
        first = jobs[0]
        for job in jobs[1:]:
            if (job["runner"], job["baseline_build"], job["candidate_build"],
                job["baseline_executable_sha256"], job["candidate_executable_sha256"],
                job["run_id"], job["attempt"], job["job_key"]) != (
                first["runner"], first["baseline_build"], first["candidate_build"],
                first["baseline_executable_sha256"], first["candidate_executable_sha256"],
                first["run_id"], first["attempt"], first["job_key"],
            ):
                reject("three jobs differ in runner, build, executable, or hosted identity")
        medians = [Decimal(job["median_ratio"]) for job in jobs]
        reference = median(medians)
        spread = (max(medians) - min(medians)) / reference
        report.update({"jobs": jobs, "between_job_spread": str(spread),
                       "reference_median_ratio": str(reference)})
        if any(Decimal(job["mad_ratio"]) > Decimal("0.10") or
               Decimal(job["p90_ratio"]) > Decimal("1.50") for job in jobs) or spread > Decimal("0.10"):
            reject("paired ratios exceed locked dispersion limits", "S225_PAIR_DISPERSION")
        if any(value > Decimal("1.25") for value in medians):
            reject("a job median candidate/baseline ratio exceeds 1.25", "S225_PAIR_REGRESSION")
        report.update({"status": "pass", "primary_rule": "S225_PAIR_PASS",
                       "detail": "three same-machine paired jobs pass the one-cell timing gate"})
        return report, 0
    except common.EvidenceError as exc:
        report.update({"primary_rule": exc.rule, "detail": exc.detail,
                       "status": "unstable" if exc.rule == "S225_PAIR_DISPERSION" else
                                 "regression" if exc.rule == "S225_PAIR_REGRESSION" else "blocked"})
        return report, 1 if exc.rule in ("S225_PAIR_DISPERSION", "S225_PAIR_REGRESSION") else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--checkout", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, code = decide(args.bundle_root, args.candidate_commit, args.checkout,
                          args.run_id, args.run_attempt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"refactor-csc-pair: {report['primary_rule']}: {report['detail']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
