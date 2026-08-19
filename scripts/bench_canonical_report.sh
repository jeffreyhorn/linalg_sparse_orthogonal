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
canonical_build_mode_override="${SPARSE_CANONICAL_BUILD_MODE:-}"
omp_num_threads="${OMP_NUM_THREADS:-unset}"
report_family="benchmark"
row_status="measurement"
support_tier="${SPARSE_CANONICAL_SUPPORT_TIER:-local_only}"
claim_boundary="${SPARSE_CANONICAL_CLAIM_BOUNDARY:-local_threshold_free}"
unselected_support_tier="local_only"
unselected_claim_boundary="local_threshold_free"
runner_context="${SPARSE_CANONICAL_RUNNER_CONTEXT:-local}"
build_flags="${SPARSE_CANONICAL_BUILD_FLAGS:-not_recorded}"
cpu_model="${SPARSE_CANONICAL_CPU_MODEL:-unknown}"
baseline="n/a"
threshold="n/a"
warmup="not_recorded"
variance="not_recorded"
matrix_size="not_recorded"
backend_context="n/a"
methodology_notes="${SPARSE_CANONICAL_METHODOLOGY_NOTES:-threshold_free_local_measurement;not_portable_performance_claim}"

reject_tsv_control_chars() {
    local field_name="$1"
    local field_value="$2"

    case "$field_value" in
        *$'\t'* | *$'\n'* | *$'\r'*)
            echo "bench-canonical-report: $field_name must not contain tabs or newlines" >&2
            exit 2
            ;;
    esac
}

reject_tsv_control_chars "BENCH_CANONICAL_REPORT_LABEL" "$report_label"
reject_tsv_control_chars "SPARSE_CANONICAL_BUILD_MODE" "$canonical_build_mode_override"
reject_tsv_control_chars "OMP_NUM_THREADS" "$omp_num_threads"
reject_tsv_control_chars "canonical report family" "$report_family"
reject_tsv_control_chars "canonical row status" "$row_status"
reject_tsv_control_chars "SPARSE_CANONICAL_SUPPORT_TIER" "$support_tier"
reject_tsv_control_chars "SPARSE_CANONICAL_CLAIM_BOUNDARY" "$claim_boundary"
reject_tsv_control_chars "canonical unselected support tier" "$unselected_support_tier"
reject_tsv_control_chars "canonical unselected claim boundary" "$unselected_claim_boundary"
reject_tsv_control_chars "SPARSE_CANONICAL_RUNNER_CONTEXT" "$runner_context"
reject_tsv_control_chars "SPARSE_CANONICAL_BUILD_FLAGS" "$build_flags"
reject_tsv_control_chars "SPARSE_CANONICAL_CPU_MODEL" "$cpu_model"
reject_tsv_control_chars "canonical baseline" "$baseline"
reject_tsv_control_chars "canonical threshold" "$threshold"
reject_tsv_control_chars "canonical warmup" "$warmup"
reject_tsv_control_chars "canonical variance" "$variance"
reject_tsv_control_chars "canonical matrix size" "$matrix_size"
reject_tsv_control_chars "canonical backend context" "$backend_context"
reject_tsv_control_chars "SPARSE_CANONICAL_METHODOLOGY_NOTES" "$methodology_notes"

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
platform="$(uname -a 2>/dev/null || echo unknown)"
cc_version="$(${CC:-cc} --version 2>/dev/null | head -n 1 || echo unknown)"

detect_openmp_runtime() {
    local binary="$1"

    if [ ! -x "$binary" ]; then
        return 1
    fi

    if command -v otool >/dev/null 2>&1; then
        if otool -L "$binary" 2>/dev/null | grep -Eiq 'lib(omp|gomp|iomp)'; then
            return 0
        fi
    fi

    if command -v ldd >/dev/null 2>&1; then
        if ldd "$binary" 2>/dev/null | grep -Eiq 'lib(omp|gomp|iomp)'; then
            return 0
        fi
    fi

    return 1
}

detect_build_mode() {
    local binary

    if [ -n "$canonical_build_mode_override" ]; then
        printf '%s\n' "$canonical_build_mode_override"
        return 0
    fi

    for binary in "$bench_refactor_csc" "$bench_chol_csc" "$bench_iterative_reuse" "$bench_eigs_reuse"; do
        if detect_openmp_runtime "$binary"; then
            printf 'openmp\n'
            return 0
        fi
    done

    printf 'serial\n'
}

build_mode="$(detect_build_mode)"

if [ -z "$git_commit" ]; then
    git_commit="unknown"
fi
if [ -z "$git_branch" ]; then
    git_branch="unknown"
elif [ "$git_branch" = "HEAD" ]; then
    git_branch="detached"
fi

{
    printf '%s\n' "surface	category	report_label	generated_at_utc	git_commit	git_branch	platform	compiler	runner_context	build_flags	cpu_model	build_mode	omp_num_threads	artifact	relative_path	command	report_family	status	support_tier	claim_boundary	fixture_or_workload	matrix_size	repeat_semantics	warmup	variance	baseline	threshold	backend_context	methodology_notes"

    emit_index_row() {
        local artifact="$1"
        local relative_path="$2"
        local command="$3"
        local fixture_or_workload="$4"
        local repeat_semantics="$5"
        local row_support_tier="$unselected_support_tier"
        local row_claim_boundary="$unselected_claim_boundary"

        if [ "$artifact" = "bench_refactor_csc" ]; then
            row_support_tier="$support_tier"
            row_claim_boundary="$claim_boundary"
        fi

        reject_tsv_control_chars "canonical artifact" "$artifact"
        reject_tsv_control_chars "canonical relative_path" "$relative_path"
        reject_tsv_control_chars "canonical command" "$command"
        reject_tsv_control_chars "canonical fixture_or_workload" "$fixture_or_workload"
        reject_tsv_control_chars "canonical repeat_semantics" "$repeat_semantics"
        reject_tsv_control_chars "canonical row support_tier" "$row_support_tier"
        reject_tsv_control_chars "canonical row claim_boundary" "$row_claim_boundary"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "canonical" "measurement" "$report_label" "$timestamp_utc" "$git_commit" "$git_branch" \
            "$platform" "$cc_version" "$runner_context" "$build_flags" "$cpu_model" "$build_mode" \
            "$omp_num_threads" "$artifact" "$relative_path" "$command" "$report_family" "$row_status" \
            "$row_support_tier" "$row_claim_boundary" "$fixture_or_workload" "$matrix_size" "$repeat_semantics" \
            "$warmup" "$variance" "$baseline" "$threshold" "$backend_context" "$methodology_notes"
    }

    emit_index_row "bench_refactor_csc" "$(basename "$refactor_csv")" \
        "tests/data/suitesparse/nos4.mtx --repeat 1" "nos4.mtx" "configured_repeat_1"
    emit_index_row "bench_chol_csc" "$(basename "$chol_csv")" \
        "tests/data/suitesparse/nos4.mtx --repeat 1" "nos4.mtx" "configured_repeat_1"
    emit_index_row "bench_iterative_reuse" "$(basename "$iter_csv")" \
        "default" "default" "benchmark_default"
    emit_index_row "bench_eigs_reuse" "$(basename "$eigs_csv")" \
        "default" "default" "benchmark_default"
} > "$index_tsv"

cat > "$manifest_txt" <<EOF
bench-canonical-report
generated_at_utc=$timestamp_utc
report_dir=$report_dir
report_label=$report_label
git_commit=$git_commit
git_branch=$git_branch
platform=$platform
compiler=$cc_version
runner_context=$runner_context
build_flags=$build_flags
cpu_model=$cpu_model
build_mode=$build_mode
omp_num_threads=$omp_num_threads

surface=canonical
category=measurement
report_family=$report_family
status=$row_status
support_tier=$support_tier
claim_boundary=$claim_boundary
selected_artifact=bench_refactor_csc
unselected_support_tier=$unselected_support_tier
unselected_claim_boundary=$unselected_claim_boundary
baseline=$baseline
threshold=$threshold
warmup=$warmup
variance=$variance
matrix_size=$matrix_size
backend_context=$backend_context
methodology_notes=$methodology_notes

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
- These rows are methodology-bound local measurement artifacts. They record the command, fixture, artifact, commit, branch, platform, compiler, build mode, and thread setting available at generation time.
- They are not portable performance guarantees, state-of-the-art claims, broad platform parity claims, package evidence, package-manager claims, shared-library or ABI guarantees, runtime-loader claims, external-library parity claims, OpenMP speedup claims, or backend superiority claims.
EOF

echo "bench-canonical-report: wrote $report_dir"
echo "  - $(basename "$refactor_csv")"
echo "  - $(basename "$chol_csv")"
echo "  - $(basename "$iter_csv")"
echo "  - $(basename "$eigs_csv")"
echo "  - $(basename "$index_tsv")"
echo "  - $(basename "$manifest_txt")"
