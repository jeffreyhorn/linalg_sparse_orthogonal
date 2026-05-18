#!/usr/bin/env python3
"""Generate and validate Sprint 33 dead-code reports."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


COVERAGE_GAP = "coverage-gap"
INTERNAL_CANDIDATE = "definitely-unused-internal-candidate"
PUBLIC_REVIEW = "public-surface-review"
SECONDARY_SIGNAL = "secondary-candidate-signal"
NOISE = "non-deadcode-static-analysis-noise"

REVIEWED_PUBLIC_KEEPS = {
    "givens_apply_right": "keep-public-api-day8-audited",
    "sparse_print_dense": "keep-public-api-day8-audited",
    "sparse_print_entries": "keep-public-api-day8-audited",
    "sparse_print_info": "keep-public-api-day8-audited",
}

XUNUSED_WARN_RE = re.compile(r"^(.+?):(\d+): warning: Function '([^']+)' is unused$")
XUNUSED_NOTE_RE = re.compile(r"^(.+?):(\d+): note: declared here$")
CPPCHECK_RE = re.compile(
    r"^(src/[^:]+):(\d+):(\d+):\s+(\w+):\s+(.*?)\s+\[([^\]]+)\]$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate dead-code reports from raw Sprint 33 artifacts."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the generated report inputs and categorizations without rewriting output.",
    )
    parser.add_argument("artifacts_dir", help="Directory containing dead-code raw artifacts.")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"deadcode_report: required artifact missing: {path}")


def parse_coverage_notes(path: Path) -> dict[str, object]:
    data: dict[str, object] = {
        "compile_commands_json": "",
        "counts": {},
        "missing_benchmarks": [],
        "missing_examples": [],
    }
    section: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "missing_benchmarks":
            section = "missing_benchmarks"
            continue
        if line == "missing_examples":
            section = "missing_examples"
            continue
        if line.startswith("compile_commands_json "):
            data["compile_commands_json"] = line.split(" ", 1)[1]
            section = None
            continue
        if section and line.startswith("- "):
            cast_list = data[section]
            assert isinstance(cast_list, list)
            cast_list.append(line[2:])
            continue
        if " " in line and section is None:
            key, value = line.split(" ", 1)
            if key in {"src", "tests", "benchmarks", "examples"}:
                counts = data["counts"]
                assert isinstance(counts, dict)
                counts[key] = int(value)
    return data


def classify_xunused(symbol: str, decl_file: str) -> tuple[str, str]:
    if symbol in REVIEWED_PUBLIC_KEEPS:
        return PUBLIC_REVIEW, REVIEWED_PUBLIC_KEEPS[symbol]
    if "/include/" in decl_file or decl_file.startswith("include/"):
        return PUBLIC_REVIEW, "needs-public-surface-audit"
    return INTERNAL_CANDIDATE, "candidate-day9-cleanup-batching"


def parse_xunused(path: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in path.read_text().splitlines():
        warn_match = XUNUSED_WARN_RE.match(raw_line)
        if warn_match:
            impl_file, impl_line, symbol = warn_match.groups()
            current = {
                "tool": "xunused",
                "symbol": symbol,
                "impl_file": impl_file,
                "impl_line": impl_line,
                "decl_file": "",
                "decl_line": "",
            }
            findings.append(current)
            continue
        note_match = XUNUSED_NOTE_RE.match(raw_line)
        if note_match and current and not current["decl_file"]:
            decl_file, decl_line = note_match.groups()
            current["decl_file"] = decl_file
            current["decl_line"] = decl_line
    for finding in findings:
        bucket, disposition = classify_xunused(finding["symbol"], finding["decl_file"])
        finding["bucket"] = bucket
        finding["disposition"] = disposition
    return findings


def parse_cppcheck(path: Path) -> tuple[list[dict[str, str]], Counter[str], Counter[tuple[str, str]]]:
    findings: list[dict[str, str]] = []
    noise_counts: Counter[str] = Counter()
    secondary_counts: Counter[tuple[str, str]] = Counter()
    for raw_line in path.read_text().splitlines():
        match = CPPCHECK_RE.match(raw_line)
        if not match:
            continue
        source, line, _column, severity, message, checker_id = match.groups()
        record = {
            "source": source,
            "line": line,
            "severity": severity,
            "message": message,
            "id": checker_id,
        }
        findings.append(record)
        if checker_id in {"unusedFunction", "staticFunction"}:
            secondary_counts[(checker_id, source)] += 1
        else:
            noise_counts[checker_id] += 1
    return findings, noise_counts, secondary_counts


def build_tsv_rows(
    coverage: dict[str, object],
    xunused: list[dict[str, str]],
    noise_counts: Counter[str],
    secondary_counts: Counter[tuple[str, str]],
) -> list[list[str]]:
    rows: list[list[str]] = []

    for symbol in coverage["missing_benchmarks"]:
        rows.append(
            [
                COVERAGE_GAP,
                "coverage-notes",
                symbol,
                "benchmarks",
                "",
                "Absent from current compile_commands.json benchmark coverage.",
                "defer-until-compile-db-expanded",
            ]
        )
    for symbol in coverage["missing_examples"]:
        rows.append(
            [
                COVERAGE_GAP,
                "coverage-notes",
                symbol,
                "examples",
                "",
                "Absent from current compile_commands.json example coverage.",
                "defer-until-compile-db-expanded",
            ]
        )

    for finding in xunused:
        detail = (
            f"Unused according to xunused; declaration at "
            f"{finding['decl_file']}:{finding['decl_line']}"
        )
        rows.append(
            [
                finding["bucket"],
                "xunused",
                finding["symbol"],
                finding["impl_file"],
                finding["impl_line"],
                detail,
                finding["disposition"],
            ]
        )

    for (checker_id, source), count in sorted(
        secondary_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    ):
        rows.append(
            [
                SECONDARY_SIGNAL,
                "cppcheck",
                "",
                source,
                "",
                f"{checker_id} count={count}",
                "summarize-only-supporting-evidence",
            ]
        )

    for checker_id, count in sorted(noise_counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            [
                NOISE,
                "cppcheck",
                "",
                "src",
                "",
                f"{checker_id} count={count}",
                "appendix-only-not-cleanup-candidate",
            ]
        )
    return rows


def write_tsv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["bucket", "tool", "symbol", "path", "line", "detail", "disposition"])
        writer.writerows(rows)


def group_rows(rows: list[list[str]]) -> dict[str, list[list[str]]]:
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)
    return grouped


def top_secondary_by_file(secondary_counts: Counter[tuple[str, str]]) -> list[tuple[str, int, str]]:
    per_file: Counter[str] = Counter()
    ids_by_file: dict[str, list[str]] = defaultdict(list)
    for (checker_id, source), count in secondary_counts.items():
        per_file[source] += count
        ids_by_file[source].append(f"{checker_id}={count}")
    results: list[tuple[str, int, str]] = []
    for source, total in per_file.most_common():
        results.append((source, total, ", ".join(sorted(ids_by_file[source]))))
    return results


def write_markdown(
    path: Path,
    coverage: dict[str, object],
    xunused: list[dict[str, str]],
    noise_counts: Counter[str],
    secondary_counts: Counter[tuple[str, str]],
    rows: list[list[str]],
) -> None:
    grouped = group_rows(rows)
    top_secondary = top_secondary_by_file(secondary_counts)
    counts = coverage["counts"]
    assert isinstance(counts, dict)
    missing_benchmarks = coverage["missing_benchmarks"]
    missing_examples = coverage["missing_examples"]
    assert isinstance(missing_benchmarks, list)
    assert isinstance(missing_examples, list)

    internal = grouped.get(INTERNAL_CANDIDATE, [])
    public = grouped.get(PUBLIC_REVIEW, [])
    secondary = grouped.get(SECONDARY_SIGNAL, [])
    noise = grouped.get(NOISE, [])

    lines: list[str] = []
    lines.append("# Sprint 33 Dead-Code Report")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- compile database: `{coverage['compile_commands_json']}`")
    lines.append(
        "- compile-db translation-unit counts: "
        f"`src={counts.get('src', 0)}` "
        f"`tests={counts.get('tests', 0)}` "
        f"`benchmarks={counts.get('benchmarks', 0)}` "
        f"`examples={counts.get('examples', 0)}`"
    )
    lines.append(
        "- raw inputs: `coverage-notes.txt`, `xunused.txt`, and `cppcheck.txt` from `make deadcode`"
    )
    lines.append("")
    lines.append("## Coverage Gaps")
    lines.append("")
    lines.append(
        "The current CMake compilation database still under-covers part of the Makefile tooling surface. "
        "Scanner silence on these paths is not evidence of absence."
    )
    lines.append("")
    if missing_benchmarks:
        lines.append("- missing benchmarks:")
        for symbol in missing_benchmarks:
            lines.append(f"  - `{symbol}`")
    if missing_examples:
        lines.append("- missing examples:")
        for symbol in missing_examples:
            lines.append(f"  - `{symbol}`")
    lines.append("")
    lines.append("## Definitely-Unused Internal Candidates")
    lines.append("")
    if internal:
        for row in internal:
            lines.append(
                f"- `{row[2]}` in `{row[3]}:{row[4]}`. {row[5]} Disposition: `{row[6]}`."
            )
    else:
        lines.append("- None currently classified in this bucket.")
    lines.append("")
    lines.append("## Public-Surface Reviewed Keeps")
    lines.append("")
    if public:
        lines.append(
            "These symbols remain in the public-surface bucket because they are exported through installed "
            "headers. The current Day 8 audit outcome for all listed rows is `keep`, not cleanup."
        )
        lines.append("")
        for row in public:
            lines.append(
                f"- `{row[2]}` in `{row[3]}:{row[4]}`. {row[5]} Disposition: `{row[6]}`."
            )
    else:
        lines.append("- None currently classified in this bucket.")
    lines.append("")
    lines.append("## Secondary `cppcheck` Candidate Signals")
    lines.append("")
    if top_secondary:
        lines.append(
            "These are supporting signals only. They stay out of the cleanup queue until a later pass confirms "
            "that they represent real dead-code opportunities rather than broad static-analysis noise."
        )
        lines.append("")
        for source, total, details in top_secondary[:10]:
            lines.append(f"- `{source}` total secondary signals: `{total}` (`{details}`).")
    else:
        lines.append("- No secondary `cppcheck` candidate signals were detected.")
    lines.append("")
    lines.append("## Deferred Noise Summary")
    lines.append("")
    if noise:
        for checker_id, count in sorted(noise_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{checker_id}`: `{count}`")
    else:
        lines.append("- No non-deadcode static-analysis noise was recorded.")
    lines.append("")
    lines.append("## Current Sprint Next-Action Queue")
    lines.append("")
    if internal:
        internal_symbols = ", ".join(f"`{row[2]}`" for row in internal)
        lines.append(f"- remaining definitely-unused internal queue: {internal_symbols}.")
    else:
        lines.append("- remaining definitely-unused internal queue: none.")
    if public:
        public_symbols = ", ".join(f"`{row[2]}`" for row in public)
        lines.append(f"- public-surface reviewed keeps: {public_symbols}.")
    else:
        lines.append("- public-surface reviewed keeps: none.")
    lines.append(
        "- `cppcheck` secondary signals remain supporting evidence only; they stay summarized for future review work, not as direct removal instructions."
    )
    path.write_text("\n".join(lines) + "\n")


def run_check(
    coverage: dict[str, object],
    xunused: list[dict[str, str]],
    report_md: Path,
    report_tsv: Path,
) -> None:
    require_file(report_md)
    require_file(report_tsv)

    missing_benchmarks = coverage["missing_benchmarks"]
    missing_examples = coverage["missing_examples"]
    assert isinstance(missing_benchmarks, list)
    assert isinstance(missing_examples, list)

    report_text = report_md.read_text()
    if "## Coverage Gaps" not in report_text:
        raise SystemExit("deadcode_report: report.md missing coverage-gap section")
    for symbol in [*missing_benchmarks, *missing_examples]:
        if f"`{symbol}`" not in report_text:
            raise SystemExit(
                f"deadcode_report: report.md missing coverage-gap item for {symbol}"
            )
    uncategorized = [
        finding["symbol"]
        for finding in xunused
        if finding.get("bucket") not in {INTERNAL_CANDIDATE, PUBLIC_REVIEW}
    ]
    if uncategorized:
        raise SystemExit(
            "deadcode_report: uncategorized xunused finding(s): "
            + ", ".join(sorted(uncategorized))
        )


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    coverage_path = artifacts_dir / "coverage-notes.txt"
    xunused_path = artifacts_dir / "xunused.txt"
    cppcheck_path = artifacts_dir / "cppcheck.txt"
    report_md = artifacts_dir / "report.md"
    report_tsv = artifacts_dir / "report.tsv"

    require_file(coverage_path)
    require_file(xunused_path)
    require_file(cppcheck_path)

    coverage = parse_coverage_notes(coverage_path)
    xunused = parse_xunused(xunused_path)
    _cppcheck_findings, noise_counts, secondary_counts = parse_cppcheck(cppcheck_path)

    rows = build_tsv_rows(coverage, xunused, noise_counts, secondary_counts)

    if not args.check:
        write_tsv(report_tsv, rows)
        write_markdown(report_md, coverage, xunused, noise_counts, secondary_counts, rows)

    run_check(coverage, xunused, report_md, report_tsv)


if __name__ == "__main__":
    main()
