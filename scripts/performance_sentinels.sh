#!/usr/bin/env bash
# performance_sentinels.sh — bounded local performance sentinel bundle.
#
# This wrapper records runtime context, runs the existing thresholded
# wall-check gate, and captures threshold-free Cholesky CSC and LDLT KKT
# benchmark rows.
# It is local regression evidence, not a portable performance claim.

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "performance-sentinels: usage: $0 <report_dir> <bench_chol_csc> <bench_refactor_csc> <bench_amd_qg> <bench_reorder> <wall_baseline>" >&2
    exit 2
fi

report_dir="$1"
bench_chol_csc="$2"
bench_refactor_csc="$3"
bench_amd_qg="$4"
bench_reorder="$5"
wall_baseline="$6"

mkdir -p "$report_dir"

report_tsv="$report_dir/sentinels.tsv"
manifest_txt="$report_dir/manifest.txt"
wall_output="$report_dir/wall_check.txt"
chol_output="$report_dir/bench_chol_csc_nos4.csv"
ldlt_output="$report_dir/bench_refactor_csc_kkt.csv"

rm -f "$wall_output" "$chol_output" "$ldlt_output"

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

    if [ -n "${SPARSE_SENTINEL_BUILD_MODE:-}" ]; then
        printf '%s\n' "$SPARSE_SENTINEL_BUILD_MODE"
        return 0
    fi

    for binary in "$bench_chol_csc" "$bench_refactor_csc" "$bench_amd_qg" "$bench_reorder"; do
        if detect_openmp_runtime "$binary"; then
            printf 'openmp\n'
            return 0
        fi
    done

    printf 'serial\n'
}

reject_tsv_control_chars() {
    local field_name="$1"
    local field_value="$2"

    case "$field_value" in
        *$'\t'* | *$'\n'* | *$'\r'*)
            echo "performance-sentinels: $field_name must not contain tabs or newlines" >&2
            exit 2
            ;;
    esac
}

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
platform="$(uname -a 2>/dev/null || echo unknown)"
cc_version="$(${CC:-cc} --version 2>/dev/null | head -n 1 || echo unknown)"
build_mode="$(detect_build_mode)"
omp_num_threads="${OMP_NUM_THREADS:-unset}"
chol_dense_backend="${SPARSE_CHOL_DENSE_BACKEND:-unset}"
ldlt_dense_backend="${SPARSE_LDLT_DENSE_BACKEND:-unset}"

reject_tsv_control_chars "SPARSE_SENTINEL_BUILD_MODE" "$build_mode"
reject_tsv_control_chars "OMP_NUM_THREADS" "$omp_num_threads"
reject_tsv_control_chars "SPARSE_CHOL_DENSE_BACKEND" "$chol_dense_backend"
reject_tsv_control_chars "SPARSE_LDLT_DENSE_BACKEND" "$ldlt_dense_backend"

if [ -z "$git_commit" ]; then
    git_commit="unknown"
fi
if [ -z "$git_branch" ]; then
    git_branch="unknown"
elif [ "$git_branch" = "HEAD" ]; then
    git_branch="detached"
fi

cat > "$report_tsv" <<EOF
report_family	sentinel_id	status	support_tier	claim_boundary	command	build_mode	omp_num_threads	matrix_or_fixture	metric	value	baseline	threshold	artifact	backend_request	backend_selected	backend_fallback	dense_kernel	panel_solver	notes
EOF

append_row() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "sentinel" "$1" "$2" "$3" "$4" "$5" "$build_mode" "$omp_num_threads" "$6" "$7" \
        "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" >> "$report_tsv"
}

wall_status="skip"
wall_note="not_run"
if [ ! -x "$bench_amd_qg" ]; then
    append_row "S5" "skip" "reviewed_thresholded" "local_wall_gate" "make wall-check" "n/a" "wall_check" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "bench_amd_qg_missing"
elif [ ! -x "$bench_reorder" ]; then
    append_row "S5" "skip" "reviewed_thresholded" "local_wall_gate" "make wall-check" "n/a" "wall_check" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "bench_reorder_missing"
elif [ ! -r "$wall_baseline" ]; then
    append_row "S5" "skip" "reviewed_thresholded" "local_wall_gate" "make wall-check" "n/a" "wall_check" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "baseline_missing"
else
    if scripts/wall_check.sh "$bench_amd_qg" "$bench_reorder" "$wall_baseline" > "$wall_output" 2>&1; then
        wall_status="pass"
        wall_note="existing_threshold_gate_passed"
    else
        wall_status="fail"
        wall_note="existing_threshold_gate_failed"
    fi

    awk -v status="$wall_status" \
        -v cmd="make wall-check" \
        -v build_mode="$build_mode" \
        -v omp="$omp_num_threads" \
        -v artifact="$(basename "$wall_output")" \
        -v note="$wall_note" '
        BEGIN { OFS="\t" }
        /^wall-check: bcsstk14/ {
            print "sentinel", "S5", status, "reviewed_thresholded", "local_wall_gate", cmd, build_mode, omp, "bcsstk14", "qg_amd_reorder_ms", $5, $8, "2x", artifact, "n/a", "n/a", "n/a", "n/a", "n/a", note
        }
        /^wall-check: Pres_Poisson AMD/ {
            print "sentinel", "S5", status, "reviewed_thresholded", "local_wall_gate", cmd, build_mode, omp, "Pres_Poisson", "amd_reorder_ms", $5, $8, "2x", artifact, "n/a", "n/a", "n/a", "n/a", "n/a", note
        }
        /^wall-check: Pres_Poisson ND/ {
            print "sentinel", "S5", status, "reviewed_thresholded", "local_wall_gate", cmd, build_mode, omp, "Pres_Poisson", "nd_reorder_ms", $5, $8, "1.5x", artifact, "n/a", "n/a", "n/a", "n/a", "n/a", note
        }
    ' "$wall_output" >> "$report_tsv"
fi

chol_cmd="$bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1"
if [ ! -x "$bench_chol_csc" ]; then
    append_row "S2" "skip" "reviewed_threshold_free" "local_threshold_free" "$chol_cmd" "nos4.mtx" "bench_chol_csc" "n/a" "n/a" "n/a" "n/a" "$chol_dense_backend" "unknown" "n/a" "unknown" "unknown" "bench_chol_csc_missing"
elif [ ! -r "tests/data/suitesparse/nos4.mtx" ]; then
    append_row "S2" "skip" "reviewed_threshold_free" "local_threshold_free" "$chol_cmd" "nos4.mtx" "bench_chol_csc" "n/a" "n/a" "n/a" "n/a" "$chol_dense_backend" "unknown" "n/a" "unknown" "unknown" "fixture_missing"
else
    if "$bench_chol_csc" tests/data/suitesparse/nos4.mtx --repeat 1 > "$chol_output"; then
        awk -F, -v cmd="$chol_cmd" \
            -v build_mode="$build_mode" \
            -v omp="$omp_num_threads" \
            -v chol_env="$chol_dense_backend" \
            -v ldlt_env="$ldlt_dense_backend" \
            -v artifact="$(basename "$chol_output")" '
            BEGIN { OFS="\t" }
            NR == 2 {
                fixture = $3
                note = "threshold_free;chol_env=" chol_env ";ldlt_env=" ldlt_env
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "factor_ll_ms", $11, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "factor_csc_ms", $12, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "factor_csc_sn_ms", $13, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_ll_ms", $14, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_csc_ms", $15, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_csc_sn_ms", $16, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "speedup_csc", $17, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "speedup_csc_sn", $18, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note
            }
        ' "$chol_output" >> "$report_tsv"
    else
        append_row "S2" "skip" "reviewed_threshold_free" "local_threshold_free" "$chol_cmd" "nos4.mtx" "bench_chol_csc" "n/a" "n/a" "n/a" "$(basename "$chol_output")" "$chol_dense_backend" "unknown" "n/a" "unknown" "unknown" "bench_run_failed"
    fi
fi

ldlt_cmd="$bench_refactor_csc --indefinite-kkt --repeat 1"
if [ ! -x "$bench_refactor_csc" ]; then
    append_row "S3" "skip" "reviewed_threshold_free" "local_threshold_free" "$ldlt_cmd" "kkt-150" "bench_refactor_csc" "n/a" "n/a" "n/a" "n/a" "$ldlt_dense_backend" "unknown" "unknown" "n/a" "n/a" "bench_refactor_csc_missing"
else
    if "$bench_refactor_csc" --indefinite-kkt --repeat 1 > "$ldlt_output"; then
        awk -F, -v cmd="$ldlt_cmd" \
            -v build_mode="$build_mode" \
            -v omp="$omp_num_threads" \
            -v env_request="$ldlt_dense_backend" \
            -v artifact="$(basename "$ldlt_output")" '
            BEGIN { OFS="\t" }
            NR == 2 {
                fixture = $3
                backend_request = $7
                backend_selected = $8
                backend_fallback = $9
                note = "threshold_free;ldlt_env=" env_request ";scenario=" $4
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "analyze_ms", $10, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "refactor_public_ms", $11, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "refactor_csc_ms", $12, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_public_ms", $13, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_csc_ms", $14, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "speedup_refactor", $15, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "res_public", $16, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "res_csc", $17, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note
            }
        ' "$ldlt_output" >> "$report_tsv"
    else
        append_row "S3" "skip" "reviewed_threshold_free" "local_threshold_free" "$ldlt_cmd" "kkt-150" "bench_refactor_csc" "n/a" "n/a" "n/a" "$(basename "$ldlt_output")" "$ldlt_dense_backend" "unknown" "unknown" "n/a" "n/a" "bench_run_failed"
    fi
fi

{
cat <<EOF
performance-sentinels
generated_at_utc=$timestamp_utc
report_dir=$report_dir
git_commit=$git_commit
git_branch=$git_branch
platform=$platform
compiler=$cc_version
build_mode=$build_mode
omp_num_threads=$omp_num_threads
sparse_chol_dense_backend=$chol_dense_backend
sparse_ldlt_dense_backend=$ldlt_dense_backend

commands:
- S5: make wall-check
- S2: $chol_cmd
- S3: $ldlt_cmd

artifacts:
- $(basename "$report_tsv")
- $(basename "$manifest_txt")
EOF

if [ -e "$wall_output" ]; then
    echo "- $(basename "$wall_output")"
fi
if [ -e "$chol_output" ]; then
    echo "- $(basename "$chol_output")"
fi
if [ -e "$ldlt_output" ]; then
    echo "- $(basename "$ldlt_output")"
fi

cat <<EOF

notes:
- S5 is the existing thresholded wall-check gate and may fail this script.
- S2 is threshold-free local reporting; compare across local runs only.
- S3 is threshold-free LDLT KKT backend reporting; compare across local runs only.
- This bundle is local regression evidence, not a portable performance claim.
EOF
} > "$manifest_txt"

echo "performance-sentinels: wrote $report_dir"
echo "  - $(basename "$report_tsv")"
echo "  - $(basename "$manifest_txt")"
if [ -e "$wall_output" ]; then
    echo "  - $(basename "$wall_output")"
fi
if [ -e "$chol_output" ]; then
    echo "  - $(basename "$chol_output")"
fi
if [ -e "$ldlt_output" ]; then
    echo "  - $(basename "$ldlt_output")"
fi

if [ "$wall_status" = "fail" ]; then
    exit 1
fi
exit 0
