#!/usr/bin/env bash
# bench_canonical_report.sh — threshold-free canonical performance snapshot
# with bounded longitudinal-report metadata.

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

report_label="${BENCH_CANONICAL_REPORT_LABEL:-}"
if [ -z "$report_label" ]; then
    report_label="unlabeled"
fi

case "$report_label" in
    *$'\t'* | *$'\n'* | *$'\r'*)
        echo "bench-canonical-report: BENCH_CANONICAL_REPORT_LABEL must not contain tabs or newlines" >&2
        exit 2
        ;;
esac

mkdir -p "$report_dir"

refactor_csv="$report_dir/bench_refactor_csc.csv"
chol_csv="$report_dir/bench_chol_csc.csv"
iter_csv="$report_dir/bench_iterative_reuse.csv"
eigs_csv="$report_dir/bench_eigs_reuse.csv"
manifest_txt="$report_dir/manifest.txt"
index_tsv="$report_dir/index.tsv"

"$bench_refactor_csc" tests/data/suitesparse/nos4.mtx --repeat 1 > "$refactor_csv"
"$bench_chol_csc" tests/data/suitesparse/nos4.mtx --repeat 1 > "$chol_csv"
"$bench_iterative_reuse" > "$iter_csv"
"$bench_eigs_reuse" > "$eigs_csv"

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
git_commit="$(git rev-parse --short HEAD 2>/dev/null || true)"
git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"

if [ -z "$git_commit" ]; then
    git_commit="unknown"
fi
if [ -z "$git_branch" ]; then
    git_branch="unknown"
elif [ "$git_branch" = "HEAD" ]; then
    git_branch="detached"
fi

cat > "$index_tsv" <<EOF
surface	category	report_label	generated_at_utc	git_commit	git_branch	artifact	relative_path	command
canonical	proof	$report_label	$timestamp_utc	$git_commit	$git_branch	bench_refactor_csc	$(basename "$refactor_csv")	tests/data/suitesparse/nos4.mtx --repeat 1
canonical	proof	$report_label	$timestamp_utc	$git_commit	$git_branch	bench_chol_csc	$(basename "$chol_csv")	tests/data/suitesparse/nos4.mtx --repeat 1
canonical	proof	$report_label	$timestamp_utc	$git_commit	$git_branch	bench_iterative_reuse	$(basename "$iter_csv")	default
canonical	proof	$report_label	$timestamp_utc	$git_commit	$git_branch	bench_eigs_reuse	$(basename "$eigs_csv")	default
EOF

cat > "$manifest_txt" <<EOF
bench-canonical-report
generated_at_utc=$timestamp_utc
report_dir=$report_dir
report_label=$report_label
git_commit=$git_commit
git_branch=$git_branch

surface=canonical
category=proof

bench_refactor_csc=tests/data/suitesparse/nos4.mtx --repeat 1 -> $(basename "$refactor_csv")
bench_chol_csc=tests/data/suitesparse/nos4.mtx --repeat 1 -> $(basename "$chol_csv")
bench_iterative_reuse=default -> $(basename "$iter_csv")
bench_eigs_reuse=default -> $(basename "$eigs_csv")
index_tsv=$(basename "$index_tsv")

artifacts:
- $(basename "$refactor_csv")
- $(basename "$chol_csv")
- $(basename "$iter_csv")
- $(basename "$eigs_csv")
- $(basename "$index_tsv")
- $(basename "$manifest_txt")

notes:
- This is a threshold-free local/CI-friendly snapshot, not a pass/fail timing gate.
- Compare CSV rows across branches or runs; do not interpret single-run timings as portable claims.
EOF

echo "bench-canonical-report: wrote $report_dir"
echo "  - $(basename "$refactor_csv")"
echo "  - $(basename "$chol_csv")"
echo "  - $(basename "$iter_csv")"
echo "  - $(basename "$eigs_csv")"
echo "  - $(basename "$index_tsv")"
echo "  - $(basename "$manifest_txt")"
