#!/usr/bin/env bash
# large_matrix_guardrails.sh — deterministic large-matrix guardrail runner.
#
# Reviewed mode runs structural tests and bounded report-shape checks only.
# Supplemental mode adds threshold-free local benchmark reports.  Supplemental
# rows are intentionally not hard timing gates.

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "large-matrix-guardrails: usage: $0 <report_dir> <test_graph> <test_reorder_nd> <test_reorder_amd_qg> <bench_reorder> <bench_amd_qg>" >&2
    exit 2
fi

report_dir="$1"
test_graph="$2"
test_reorder_nd="$3"
test_reorder_amd_qg="$4"
bench_reorder="$5"
bench_amd_qg="$6"

supplemental="${SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL:-0}"

case "$supplemental" in
    0 | 1) ;;
    *)
        echo "large-matrix-guardrails: SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL must be 0 or 1" >&2
        exit 2
        ;;
esac

mkdir -p "$report_dir"

manifest_txt="$report_dir/manifest.txt"
index_tsv="$report_dir/index.tsv"
graph_out="$report_dir/test_graph.txt"
nd_out="$report_dir/test_reorder_nd.txt"
amd_qg_out="$report_dir/test_reorder_amd_qg.txt"
reorder_sprint86_csv="$report_dir/bench_reorder_sprint86.csv"
reorder_all_csv="$report_dir/bench_reorder_all.csv"
amd_qg_csv="$report_dir/bench_amd_qg_skip_bitset.csv"

timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
git_commit="$(git rev-parse --short HEAD 2>/dev/null || true)"
git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
platform_string="$(uname -a 2>/dev/null || true)"
compiler_string="$("${CC:-cc}" --version 2>/dev/null | sed -n '1p' || true)"

if [ -z "$git_commit" ]; then
    git_commit="unknown"
fi
if [ -z "$git_branch" ]; then
    git_branch="unknown"
elif [ "$git_branch" = "HEAD" ]; then
    git_branch="detached"
fi
if [ -z "$platform_string" ]; then
    platform_string="unknown"
fi
if [ -z "$compiler_string" ]; then
    compiler_string="unknown"
fi

require_executable() {
    if [ ! -x "$1" ]; then
        echo "large-matrix-guardrails: $1 not executable" >&2
        exit 2
    fi
}

require_executable "$test_graph"
require_executable "$test_reorder_nd"
require_executable "$test_reorder_amd_qg"
require_executable "$bench_reorder"
require_executable "$bench_amd_qg"

write_index_header() {
    cat > "$index_tsv" <<EOF
lane_id	status	category	command	artifact	notes
EOF
}

append_index_row() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$index_tsv"
}

validate_reorder_csv() {
    local csv_file="$1"
    local expected_slice="$2"
    local expected_rows="$3"
    local fixture_names="$4"

    awk -F, -v expected_slice="$expected_slice" \
        -v expected_rows="$expected_rows" \
        -v fixture_names="$fixture_names" '
        NR == 1 {
            expected = "matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold"
            if ($0 != expected) {
                printf("unexpected header: %s\n", $0) > "/dev/stderr"
                exit 1
            }
            next
        }
        {
            row_count++
            seen[$1 "," $3] = 1
            fixtures[$1] = 1
            if ($6 != "skip" || $7 != "direct" || $8 != expected_slice || $9 != "160") {
                printf("unexpected row metadata: %s\n", $0) > "/dev/stderr"
                exit 1
            }
        }
        END {
            split(fixture_names, f, " ")
            split("none rcm amd colamd nd", r, " ")
            if (row_count != expected_rows) {
                printf("unexpected row count: %d\n", row_count) > "/dev/stderr"
                exit 1
            }
            for (i in f) {
                if (!(f[i] in fixtures)) {
                    printf("missing fixture: %s\n", f[i]) > "/dev/stderr"
                    exit 1
                }
                for (j in r) {
                    key = f[i] "," r[j]
                    if (!(key in seen)) {
                        printf("missing row: %s\n", key) > "/dev/stderr"
                        exit 1
                    }
                }
            }
        }
    ' "$csv_file"
}

write_index_header

"$test_graph" > "$graph_out"
append_index_row "G3" "pass" "reviewed" "$test_graph" "$(basename "$graph_out")" "graph partition, separator, generated-family structural tests"

"$test_reorder_nd" > "$nd_out"
append_index_row "G2" "pass" "reviewed" "$test_reorder_nd" "$(basename "$nd_out")" "ND generated-family and named-matrix structural tests; explicit skips remain in artifact"

"$test_reorder_amd_qg" > "$amd_qg_out"
append_index_row "G1" "pass" "reviewed" "$test_reorder_amd_qg" "$(basename "$amd_qg_out")" "qg-AMD wrapper and banded-n10000-bw5 structural guardrail"

"$bench_reorder" --sprint86-slice --skip-factor > "$reorder_sprint86_csv"
validate_reorder_csv "$reorder_sprint86_csv" "sprint86" 10 "bcsstk14 Pres_Poisson"
append_index_row "G4" "pass" "reviewed" "$bench_reorder --sprint86-slice --skip-factor" "$(basename "$reorder_sprint86_csv")" "bounded bench_reorder CSV shape and structural fill rows"

if [ "$supplemental" = "1" ]; then
    "$bench_reorder" --skip-factor > "$reorder_all_csv"
    validate_reorder_csv "$reorder_all_csv" "all" 30 "nos4 bcsstk04 Kuu bcsstk14 s3rmt3m3 Pres_Poisson"
    append_index_row "S1" "report" "supplemental" "$bench_reorder --skip-factor" "$(basename "$reorder_all_csv")" "threshold-free full named-matrix reorder report"

    "$bench_amd_qg" --skip-bitset > "$amd_qg_csv"
    append_index_row "S2" "report" "supplemental" "$bench_amd_qg --skip-bitset" "$(basename "$amd_qg_csv")" "threshold-free qg-AMD and generated-banded report; max-RSS is platform-local"
else
    append_index_row "S1" "skip" "supplemental" "$bench_reorder --skip-factor" "n/a" "set SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 to run"
    append_index_row "S2" "skip" "supplemental" "$bench_amd_qg --skip-bitset" "n/a" "set SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 to run"
fi

cat > "$manifest_txt" <<EOF
large-matrix-guardrails
generated_at_utc=$timestamp_utc
report_dir=$report_dir
git_commit=$git_commit
git_branch=$git_branch
platform=$platform_string
compiler=$compiler_string
supplemental=$supplemental

reviewed_lanes:
- G1: $test_reorder_amd_qg
- G2: $test_reorder_nd
- G3: $test_graph
- G4: $bench_reorder --sprint86-slice --skip-factor

supplemental_lanes:
- S1: $bench_reorder --skip-factor
- S2: $bench_amd_qg --skip-bitset

artifacts:
- $(basename "$index_tsv")
- $(basename "$manifest_txt")
- $(basename "$graph_out")
- $(basename "$nd_out")
- $(basename "$amd_qg_out")
- $(basename "$reorder_sprint86_csv")
EOF

if [ "$supplemental" = "1" ]; then
    {
        echo "- $(basename "$reorder_all_csv")"
        echo "- $(basename "$amd_qg_csv")"
    } >> "$manifest_txt"
fi

cat >> "$manifest_txt" <<EOF

notes:
- Reviewed lanes are deterministic structural checks or bounded CSV shape checks.
- Supplemental lanes are threshold-free report context and are opt-in.
- This target does not add portable timing or max-RSS thresholds.
EOF

echo "large-matrix-guardrails: wrote $report_dir"
echo "  - $(basename "$index_tsv")"
echo "  - $(basename "$manifest_txt")"
echo "  - $(basename "$graph_out")"
echo "  - $(basename "$nd_out")"
echo "  - $(basename "$amd_qg_out")"
echo "  - $(basename "$reorder_sprint86_csv")"
if [ "$supplemental" = "1" ]; then
    echo "  - $(basename "$reorder_all_csv")"
    echo "  - $(basename "$amd_qg_csv")"
fi
