#!/usr/bin/env bash
# bench_canonical_report.sh — Sprint 65 Day 11 threshold-free canonical
# performance snapshot.

set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "bench-canonical-report: usage: $0 <report_dir> <bench_refactor_csc> <bench_chol_csc> <bench_iterative_reuse> <bench_eigs_reuse>" >&2
    exit 2
fi

report_dir="$1"
bench_refactor_csc="$2"
bench_chol_csc="$3"
bench_iterative_reuse="$4"
bench_eigs_reuse="$5"

mkdir -p "$report_dir"

refactor_csv="$report_dir/bench_refactor_csc.csv"
chol_csv="$report_dir/bench_chol_csc.csv"
iter_csv="$report_dir/bench_iterative_reuse.csv"
eigs_csv="$report_dir/bench_eigs_reuse.csv"
manifest_txt="$report_dir/manifest.txt"

"$bench_refactor_csc" tests/data/suitesparse/nos4.mtx --repeat 1 > "$refactor_csv"
"$bench_chol_csc" tests/data/suitesparse/nos4.mtx --repeat 1 > "$chol_csv"
"$bench_iterative_reuse" > "$iter_csv"
"$bench_eigs_reuse" > "$eigs_csv"

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cat > "$manifest_txt" <<EOF
bench-canonical-report
generated_at_utc=$timestamp_utc
report_dir=$report_dir

surface=canonical
category=proof

bench_refactor_csc=tests/data/suitesparse/nos4.mtx --repeat 1 -> $(basename "$refactor_csv")
bench_chol_csc=tests/data/suitesparse/nos4.mtx --repeat 1 -> $(basename "$chol_csv")
bench_iterative_reuse=default -> $(basename "$iter_csv")
bench_eigs_reuse=default -> $(basename "$eigs_csv")

notes:
- This is a threshold-free local/CI-friendly snapshot, not a pass/fail timing gate.
- Compare CSV rows across branches or runs; do not interpret single-run timings as portable claims.
EOF

echo "bench-canonical-report: wrote $report_dir"
echo "  - $(basename "$refactor_csv")"
echo "  - $(basename "$chol_csv")"
echo "  - $(basename "$iter_csv")"
echo "  - $(basename "$eigs_csv")"
echo "  - $(basename "$manifest_txt")"
