#!/usr/bin/env bash
# performance_sentinels.sh — bounded local performance sentinel bundle.
#
# This wrapper records runtime context, runs the existing thresholded
# wall-check gate, captures threshold-free Cholesky CSC and LDLT KKT
# benchmark rows, and checks the selected refactor CSC lane with a broad local
# smoke ceiling.
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
selected_refactor_output="$report_dir/bench_refactor_csc_nos4.csv"
threshold_free_baseline_provenance="n/a"
s6_baseline_provenance="sprint169_selected_nos4_local_smoke_ceiling"
s6_refactor_csc_ms_ceiling="${SPARSE_SELECTED_REFACTOR_CSC_MS_CEILING:-500.0}"
s5_repeat_semantics="wall_check_configured_single_runs"
s2_repeat_semantics="configured_repeat_1"
s3_repeat_semantics="configured_repeat_1"
s6_repeat_semantics="configured_repeat_1"
warmup="not_recorded"
variance="not_recorded"
s6_warmup="none_configured"
s6_variance="not_computed_single_sample"
s5_methodology_notes="thresholded_local_wall_gate;not_portable_performance_claim"
s2_methodology_notes="threshold_free_local_backend_context;not_backend_superiority_claim"
s3_methodology_notes="threshold_free_local_ldlt_backend_context;not_backend_superiority_claim"
s6_methodology_notes="selected_local_large_regression_gate;not_portable_performance_claim"

rm -f "$wall_output" "$chol_output" "$ldlt_output" "$selected_refactor_output"

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
reject_tsv_control_chars "wall-check baseline provenance" "$wall_baseline"
reject_tsv_control_chars "threshold-free baseline provenance" "$threshold_free_baseline_provenance"
reject_tsv_control_chars "S6 baseline provenance" "$s6_baseline_provenance"
reject_tsv_control_chars "S6 refactor CSC ceiling" "$s6_refactor_csc_ms_ceiling"
reject_tsv_control_chars "S5 repeat semantics" "$s5_repeat_semantics"
reject_tsv_control_chars "S2 repeat semantics" "$s2_repeat_semantics"
reject_tsv_control_chars "S3 repeat semantics" "$s3_repeat_semantics"
reject_tsv_control_chars "S6 repeat semantics" "$s6_repeat_semantics"
reject_tsv_control_chars "sentinel warmup" "$warmup"
reject_tsv_control_chars "sentinel variance" "$variance"
reject_tsv_control_chars "S6 warmup" "$s6_warmup"
reject_tsv_control_chars "S6 variance" "$s6_variance"
reject_tsv_control_chars "S5 methodology notes" "$s5_methodology_notes"
reject_tsv_control_chars "S2 methodology notes" "$s2_methodology_notes"
reject_tsv_control_chars "S3 methodology notes" "$s3_methodology_notes"
reject_tsv_control_chars "S6 methodology notes" "$s6_methodology_notes"

if ! awk -v value="$s6_refactor_csc_ms_ceiling" \
    'BEGIN { exit !((value + 0) > 0 && value ~ /^[0-9]+([.][0-9]+)?$/) }'; then
    echo "performance-sentinels: SPARSE_SELECTED_REFACTOR_CSC_MS_CEILING must be a positive numeric millisecond ceiling" >&2
    exit 2
fi

if [ -z "$git_commit" ]; then
    git_commit="unknown"
fi
if [ -z "$git_branch" ]; then
    git_branch="unknown"
elif [ "$git_branch" = "HEAD" ]; then
    git_branch="detached"
fi

cat > "$report_tsv" <<EOF
report_family	sentinel_id	status	support_tier	claim_boundary	command	build_mode	omp_num_threads	matrix_or_fixture	metric	value	baseline	threshold	artifact	backend_request	backend_selected	backend_fallback	dense_kernel	panel_solver	notes	baseline_provenance	repeat_semantics	warmup	variance	methodology_notes
EOF

append_row() {
    local baseline_provenance="${18:-n/a}"
    local repeat_semantics="${19:-not_recorded}"
    local row_warmup="${20:-$warmup}"
    local row_variance="${21:-$variance}"
    local methodology_notes="${22:-${17}}"

    reject_tsv_control_chars "sentinel notes" "${17}"
    reject_tsv_control_chars "sentinel baseline provenance" "$baseline_provenance"
    reject_tsv_control_chars "sentinel repeat semantics" "$repeat_semantics"
    reject_tsv_control_chars "sentinel row warmup" "$row_warmup"
    reject_tsv_control_chars "sentinel row variance" "$row_variance"
    reject_tsv_control_chars "sentinel methodology notes" "$methodology_notes"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "sentinel" "$1" "$2" "$3" "$4" "$5" "$build_mode" "$omp_num_threads" "$6" "$7" \
        "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" \
        "$baseline_provenance" "$repeat_semantics" "$row_warmup" "$row_variance" \
        "$methodology_notes" >> "$report_tsv"
}

wall_status="skip"
wall_note="not_run"
if [ ! -x "$bench_amd_qg" ]; then
    append_row "S5" "skip" "reviewed_thresholded" "local_wall_gate" "make wall-check" "n/a" "wall_check" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "bench_amd_qg_missing" "$wall_baseline" "$s5_repeat_semantics" "$warmup" "$variance" "$s5_methodology_notes"
elif [ ! -x "$bench_reorder" ]; then
    append_row "S5" "skip" "reviewed_thresholded" "local_wall_gate" "make wall-check" "n/a" "wall_check" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "bench_reorder_missing" "$wall_baseline" "$s5_repeat_semantics" "$warmup" "$variance" "$s5_methodology_notes"
elif [ ! -r "$wall_baseline" ]; then
    append_row "S5" "skip" "reviewed_thresholded" "local_wall_gate" "make wall-check" "n/a" "wall_check" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "baseline_missing" "$wall_baseline" "$s5_repeat_semantics" "$warmup" "$variance" "$s5_methodology_notes"
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
        -v baseline_provenance="$wall_baseline" \
        -v repeat_semantics="$s5_repeat_semantics" \
        -v warmup="$warmup" \
        -v variance="$variance" \
        -v methodology_notes="$s5_methodology_notes" \
        -v note="$wall_note" '
        BEGIN { OFS="\t" }
        /^wall-check: bcsstk14/ {
            print "sentinel", "S5", status, "reviewed_thresholded", "local_wall_gate", cmd, build_mode, omp, "bcsstk14", "qg_amd_reorder_ms", $5, $8, "2x", artifact, "n/a", "n/a", "n/a", "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
        }
        /^wall-check: Pres_Poisson AMD/ {
            print "sentinel", "S5", status, "reviewed_thresholded", "local_wall_gate", cmd, build_mode, omp, "Pres_Poisson", "amd_reorder_ms", $5, $8, "2x", artifact, "n/a", "n/a", "n/a", "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
        }
        /^wall-check: Pres_Poisson ND/ {
            print "sentinel", "S5", status, "reviewed_thresholded", "local_wall_gate", cmd, build_mode, omp, "Pres_Poisson", "nd_reorder_ms", $5, $8, "1.5x", artifact, "n/a", "n/a", "n/a", "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
        }
    ' "$wall_output" >> "$report_tsv"
fi

selected_refactor_status="skip"
selected_refactor_note="not_run"
selected_refactor_cmd="$bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1"
if [ ! -x "$bench_refactor_csc" ]; then
    append_row "S6" "skip" "reviewed_thresholded" "local_selected_regression_gate" "$selected_refactor_cmd" "nos4.mtx" "refactor_csc_ms" "n/a" "n/a" "$s6_refactor_csc_ms_ceiling" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "bench_refactor_csc_missing" "$s6_baseline_provenance" "$s6_repeat_semantics" "$s6_warmup" "$s6_variance" "$s6_methodology_notes"
elif [ ! -r "tests/data/suitesparse/nos4.mtx" ]; then
    append_row "S6" "skip" "reviewed_thresholded" "local_selected_regression_gate" "$selected_refactor_cmd" "nos4.mtx" "refactor_csc_ms" "n/a" "n/a" "$s6_refactor_csc_ms_ceiling" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "fixture_missing" "$s6_baseline_provenance" "$s6_repeat_semantics" "$s6_warmup" "$s6_variance" "$s6_methodology_notes"
elif "$bench_refactor_csc" tests/data/suitesparse/nos4.mtx --repeat 1 > "$selected_refactor_output"; then
    selected_refactor_ms="$(awk -F, 'NR == 2 && $1 == "bench_refactor_csc" && $3 == "nos4.mtx" && $4 == "chol_spd" { print $12; exit }' "$selected_refactor_output")"
    if [ -z "$selected_refactor_ms" ] || ! awk -v value="$selected_refactor_ms" \
        'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/) }'; then
        selected_refactor_status="fail"
        selected_refactor_note="selected_refactor_csc_parse_failed"
        append_row "S6" "fail" "reviewed_thresholded" "local_selected_regression_gate" "$selected_refactor_cmd" "nos4.mtx" "refactor_csc_ms" "n/a" "n/a" "$s6_refactor_csc_ms_ceiling" "$(basename "$selected_refactor_output")" "n/a" "n/a" "n/a" "n/a" "n/a" "$selected_refactor_note" "$s6_baseline_provenance" "$s6_repeat_semantics" "$s6_warmup" "$s6_variance" "$s6_methodology_notes"
        echo "performance-sentinels: FAIL S6 could not parse selected refactor_csc_ms from $(basename "$selected_refactor_output")" >&2
    elif awk -v actual="$selected_refactor_ms" -v threshold="$s6_refactor_csc_ms_ceiling" \
        'BEGIN { exit !((actual + 0) > threshold) }'; then
        selected_refactor_status="fail"
        selected_refactor_note="selected_local_smoke_ceiling_failed"
        append_row "S6" "fail" "reviewed_thresholded" "local_selected_regression_gate" "$selected_refactor_cmd" "nos4.mtx" "refactor_csc_ms" "$selected_refactor_ms" "$s6_refactor_csc_ms_ceiling" "$s6_refactor_csc_ms_ceiling" "$(basename "$selected_refactor_output")" "n/a" "n/a" "n/a" "n/a" "n/a" "$selected_refactor_note" "$s6_baseline_provenance" "$s6_repeat_semantics" "$s6_warmup" "$s6_variance" "$s6_methodology_notes"
        echo "performance-sentinels: FAIL S6 selected refactor_csc_ms=$selected_refactor_ms ms > $s6_refactor_csc_ms_ceiling ms local smoke ceiling for nos4.mtx --repeat 1" >&2
    else
        selected_refactor_status="pass"
        selected_refactor_note="selected_local_smoke_ceiling_passed"
        append_row "S6" "pass" "reviewed_thresholded" "local_selected_regression_gate" "$selected_refactor_cmd" "nos4.mtx" "refactor_csc_ms" "$selected_refactor_ms" "$s6_refactor_csc_ms_ceiling" "$s6_refactor_csc_ms_ceiling" "$(basename "$selected_refactor_output")" "n/a" "n/a" "n/a" "n/a" "n/a" "$selected_refactor_note" "$s6_baseline_provenance" "$s6_repeat_semantics" "$s6_warmup" "$s6_variance" "$s6_methodology_notes"
    fi
else
    append_row "S6" "skip" "reviewed_thresholded" "local_selected_regression_gate" "$selected_refactor_cmd" "nos4.mtx" "refactor_csc_ms" "n/a" "n/a" "$s6_refactor_csc_ms_ceiling" "$(basename "$selected_refactor_output")" "n/a" "n/a" "n/a" "n/a" "n/a" "bench_run_failed" "$s6_baseline_provenance" "$s6_repeat_semantics" "$s6_warmup" "$s6_variance" "$s6_methodology_notes"
fi

chol_cmd="$bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1"
if [ ! -x "$bench_chol_csc" ]; then
    append_row "S2" "skip" "reviewed_threshold_free" "local_threshold_free" "$chol_cmd" "nos4.mtx" "bench_chol_csc" "n/a" "n/a" "n/a" "n/a" "$chol_dense_backend" "unknown" "n/a" "unknown" "unknown" "bench_chol_csc_missing" "$threshold_free_baseline_provenance" "$s2_repeat_semantics" "$warmup" "$variance" "$s2_methodology_notes"
elif [ ! -r "tests/data/suitesparse/nos4.mtx" ]; then
    append_row "S2" "skip" "reviewed_threshold_free" "local_threshold_free" "$chol_cmd" "nos4.mtx" "bench_chol_csc" "n/a" "n/a" "n/a" "n/a" "$chol_dense_backend" "unknown" "n/a" "unknown" "unknown" "fixture_missing" "$threshold_free_baseline_provenance" "$s2_repeat_semantics" "$warmup" "$variance" "$s2_methodology_notes"
else
    if "$bench_chol_csc" tests/data/suitesparse/nos4.mtx --repeat 1 > "$chol_output"; then
        awk -F, -v cmd="$chol_cmd" \
            -v build_mode="$build_mode" \
            -v omp="$omp_num_threads" \
            -v chol_env="$chol_dense_backend" \
            -v ldlt_env="$ldlt_dense_backend" \
            -v baseline_provenance="$threshold_free_baseline_provenance" \
            -v repeat_semantics="$s2_repeat_semantics" \
            -v warmup="$warmup" \
            -v variance="$variance" \
            -v methodology_notes="$s2_methodology_notes" \
            -v artifact="$(basename "$chol_output")" '
            BEGIN { OFS="\t" }
            NR == 2 {
                fixture = $3
                note = "threshold_free;chol_env=" chol_env ";ldlt_env=" ldlt_env
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "factor_ll_ms", $11, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "factor_csc_ms", $12, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "factor_csc_sn_ms", $13, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_ll_ms", $14, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_csc_ms", $15, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_csc_sn_ms", $16, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "speedup_csc", $17, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S2", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "speedup_csc_sn", $18, "n/a", "n/a", artifact, chol_env, $9, "n/a", $9, $10, note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
            }
        ' "$chol_output" >> "$report_tsv"
    else
        append_row "S2" "skip" "reviewed_threshold_free" "local_threshold_free" "$chol_cmd" "nos4.mtx" "bench_chol_csc" "n/a" "n/a" "n/a" "$(basename "$chol_output")" "$chol_dense_backend" "unknown" "n/a" "unknown" "unknown" "bench_run_failed" "$threshold_free_baseline_provenance" "$s2_repeat_semantics" "$warmup" "$variance" "$s2_methodology_notes"
    fi
fi

ldlt_cmd="$bench_refactor_csc --indefinite-kkt --repeat 1"
if [ ! -x "$bench_refactor_csc" ]; then
    append_row "S3" "skip" "reviewed_threshold_free" "local_threshold_free" "$ldlt_cmd" "kkt-150" "bench_refactor_csc" "n/a" "n/a" "n/a" "n/a" "$ldlt_dense_backend" "unknown" "unknown" "n/a" "n/a" "bench_refactor_csc_missing" "$threshold_free_baseline_provenance" "$s3_repeat_semantics" "$warmup" "$variance" "$s3_methodology_notes"
else
    if "$bench_refactor_csc" --indefinite-kkt --repeat 1 > "$ldlt_output"; then
        awk -F, -v cmd="$ldlt_cmd" \
            -v build_mode="$build_mode" \
            -v omp="$omp_num_threads" \
            -v env_request="$ldlt_dense_backend" \
            -v baseline_provenance="$threshold_free_baseline_provenance" \
            -v repeat_semantics="$s3_repeat_semantics" \
            -v warmup="$warmup" \
            -v variance="$variance" \
            -v methodology_notes="$s3_methodology_notes" \
            -v artifact="$(basename "$ldlt_output")" '
            BEGIN { OFS="\t" }
            NR == 2 {
                fixture = $3
                backend_request = $7
                backend_selected = $8
                backend_fallback = $9
                note = "threshold_free;ldlt_env=" env_request ";scenario=" $4
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "analyze_ms", $10, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "refactor_public_ms", $11, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "refactor_csc_ms", $12, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_public_ms", $13, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "solve_csc_ms", $14, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "speedup_refactor", $15, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "res_public", $16, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
                print "sentinel", "S3", "report", "reviewed_threshold_free", "local_threshold_free", cmd, build_mode, omp, fixture, "res_csc", $17, "n/a", "n/a", artifact, backend_request, backend_selected, backend_fallback, "n/a", "n/a", note, baseline_provenance, repeat_semantics, warmup, variance, methodology_notes
            }
        ' "$ldlt_output" >> "$report_tsv"
    else
        append_row "S3" "skip" "reviewed_threshold_free" "local_threshold_free" "$ldlt_cmd" "kkt-150" "bench_refactor_csc" "n/a" "n/a" "n/a" "$(basename "$ldlt_output")" "$ldlt_dense_backend" "unknown" "unknown" "n/a" "n/a" "bench_run_failed" "$threshold_free_baseline_provenance" "$s3_repeat_semantics" "$warmup" "$variance" "$s3_methodology_notes"
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
s5_baseline_provenance=$wall_baseline
s6_baseline_provenance=$s6_baseline_provenance
s6_refactor_csc_ms_ceiling=$s6_refactor_csc_ms_ceiling
s5_repeat_semantics=$s5_repeat_semantics
s2_repeat_semantics=$s2_repeat_semantics
s3_repeat_semantics=$s3_repeat_semantics
s6_repeat_semantics=$s6_repeat_semantics
warmup=$warmup
variance=$variance
s6_warmup=$s6_warmup
s6_variance=$s6_variance

commands:
- S5: make wall-check
- S6: $selected_refactor_cmd
- S2: $chol_cmd
- S3: $ldlt_cmd

artifacts:
- $(basename "$report_tsv")
- $(basename "$manifest_txt")
EOF

if [ -e "$wall_output" ]; then
    echo "- $(basename "$wall_output")"
fi
if [ -e "$selected_refactor_output" ]; then
    echo "- $(basename "$selected_refactor_output")"
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
- S6 is a broad local selected-lane smoke ceiling for bench_refactor_csc on nos4.mtx --repeat 1 and may fail this script.
- S2 is threshold-free local reporting; compare across local runs only.
- S3 is threshold-free LDLT KKT backend reporting; compare across local runs only.
- This bundle is local regression evidence, not a portable performance claim.
- S5 status is meaningful only with the recorded baseline, threshold, fixture, command, and machine context. It is not a portable timing promise.
- S6 status is meaningful only with the selected fixture, command, local smoke ceiling, build mode, OMP_NUM_THREADS, and machine context. It is not a portable timing promise or hosted publication claim.
- S2 and S3 rows are threshold-free local backend-context rows. They preserve backend request, selected backend, fallback, dense-kernel, and panel-solver context where emitted, but they do not pass or fail and do not prove backend superiority.
- These rows are not state-of-the-art claims, broad platform parity claims, package evidence, package-manager claims, shared-library or ABI guarantees, runtime-loader claims, external-library parity claims, OpenMP speedup claims, or backend superiority claims.
EOF
} > "$manifest_txt"

echo "performance-sentinels: wrote $report_dir"
echo "  - $(basename "$report_tsv")"
echo "  - $(basename "$manifest_txt")"
if [ -e "$wall_output" ]; then
    echo "  - $(basename "$wall_output")"
fi
if [ -e "$selected_refactor_output" ]; then
    echo "  - $(basename "$selected_refactor_output")"
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
if [ "$selected_refactor_status" = "fail" ]; then
    exit 1
fi
exit 0
