#!/usr/bin/env bash
# check_ldlt_csc_helper_guard.sh - Sprint 185 LDLT CSC helper guard.
#
# Keeps the extracted family-local helper headers tied to the registered
# `test_ldlt_csc` proof-owner binary. The headers are intentionally included
# by `tests/test_ldlt_csc.c`, not registered as standalone tests or library
# sources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_FILE="$ROOT_DIR/tests/test_ldlt_csc.c"
MAKEFILE="$ROOT_DIR/Makefile"
CMAKE_FILE="$ROOT_DIR/CMakeLists.txt"
LIBRARY_MANIFEST="$ROOT_DIR/build-metadata/library_sources.txt"
HELPERS=(
    "tests/test_ldlt_csc_fixtures.h"
    "tests/test_ldlt_csc_native_parity_helpers.h"
    "tests/test_ldlt_csc_oracle_helpers.h"
    "tests/test_ldlt_csc_supernode_helpers.h"
)

RUN_TEST_MARKERS=(
    "RUN_TEST(test_native_1x1_diagonal_matches_wrapper);"
    "RUN_TEST(test_native_1x1_tridiagonal_matches_wrapper);"
    "RUN_TEST(test_native_1x1_mixed_indefinite_matches_wrapper);"
    "RUN_TEST(test_native_1x1_with_swap_matches_wrapper);"
    "RUN_TEST(test_native_1x1_tridiag_large_matches_wrapper);"
    "RUN_TEST(test_native_detects_near_zero_1x1_pivot);"
    "RUN_TEST(test_native_1x1_identity_matches_wrapper);"
    "RUN_TEST(test_native_2x2_forced_matches_wrapper);"
    "RUN_TEST(test_native_2x2_nonadjacent_partner_matches_wrapper);"
    "RUN_TEST(test_native_mixed_pivots_matches_wrapper);"
    "RUN_TEST(test_native_mixed_pivots_larger_matches_wrapper);"
    "RUN_TEST(test_native_2x2_solve_matches_linked_list);"
    "RUN_TEST(test_native_2x2_inertia_matches_wrapper);"
)

SOLVE_RUN_TEST_MARKER="RUN_TEST(test_solve_null_args);"

MOVED_DEFINITION_MARKERS=(
    "static void test_native_1x1_diagonal_matches_wrapper(void) {"
    "static void test_native_1x1_tridiagonal_matches_wrapper(void) {"
    "static void test_native_1x1_mixed_indefinite_matches_wrapper(void) {"
    "static void test_native_1x1_with_swap_matches_wrapper(void) {"
    "static void test_native_1x1_tridiag_large_matches_wrapper(void) {"
    "static void test_native_detects_near_zero_1x1_pivot(void) {"
    "static void test_native_1x1_identity_matches_wrapper(void) {"
    "static void test_native_2x2_forced_matches_wrapper(void) {"
    "static void test_native_2x2_nonadjacent_partner_matches_wrapper(void) {"
    "static void test_native_mixed_pivots_matches_wrapper(void) {"
    "static void test_native_mixed_pivots_larger_matches_wrapper(void) {"
    "static void test_native_2x2_solve_matches_linked_list(void) {"
    "static void test_native_2x2_inertia_matches_wrapper(void) {"
)

NATIVE_HELPER="tests/test_ldlt_csc_native_parity_helpers.h"
NATIVE_HELPER_PATH="$ROOT_DIR/$NATIVE_HELPER"

fail() {
    echo "ldlt-csc-helper-guard: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "ldlt-csc-helper-guard: $1 ok"
}

require_file() {
    local path="$1"
    local message="$2"

    if [ ! -f "$path" ]; then
        fail "$message"
    fi
}

require_fixed() {
    local needle="$1"
    local file="$2"
    local message="$3"

    if ! grep -Fq "$needle" "$file"; then
        fail "$message"
    fi
}

require_absent_fixed() {
    local needle="$1"
    local file="$2"
    local message="$3"
    local matches

    matches="$(grep --fixed-strings --line-number -- "$needle" "$file" 2>/dev/null || true)"
    if [ -n "$matches" ]; then
        echo "$matches" >&2
        fail "$message"
    fi
}

require_exact_fixed_count() {
    local needle="$1"
    local file="$2"
    local expected="$3"
    local message="$4"
    local count

    count="$(grep --fixed-strings --count -- "$needle" "$file" 2>/dev/null || true)"
    if [ -z "$count" ]; then
        count=0
    fi
    if [ "$count" -ne "$expected" ]; then
        fail "$message (expected $expected, found $count)"
    fi
}

fixed_occurrence_count() {
    local needle="$1"
    local file="$2"

    awk -v needle="$needle" '
        {
            line = $0
            while ((pos = index(line, needle)) > 0) {
                count++
                line = substr(line, pos + length(needle))
            }
        }
        END { print count + 0 }
    ' "$file"
}

ACTIVE_CODE_AWK="$(cat <<'AWK'
        function if_directive(line) {
            return line ~ /^[[:space:]]*#[[:space:]]*(if|ifdef|ifndef)([[:space:]]|$)/
        }
        function elif_directive(line) {
            return line ~ /^[[:space:]]*#[[:space:]]*elif([[:space:]]|$)/
        }
        function else_directive(line) {
            return line ~ /^[[:space:]]*#[[:space:]]*else([[:space:]]|$)/
        }
        function endif_directive(line) {
            return line ~ /^[[:space:]]*#[[:space:]]*endif([[:space:]]|$)/
        }
        function condition_expression(line,    expr) {
            expr = line
            sub(/^[[:space:]]*#[[:space:]]*(if|elif)[[:space:]]+/, "", expr)
            sub(/[[:space:]]*$/, "", expr)
            gsub(/[[:space:]]+/, "", expr)
            return expr
        }
        function zero_expression(expr) {
            return expr ~ /^\(*0([xX]0+|[0]*)([uUlL]*)\)*$/
        }
        function one_expression(expr) {
            return expr ~ /^\(*1([uUlL]*)\)*$/
        }
        function false_condition(line) {
            return line ~ /^[[:space:]]*#[[:space:]]*(if|elif)[[:space:]]+/ &&
                zero_expression(condition_expression(line))
        }
        function true_condition(line) {
            return line ~ /^[[:space:]]*#[[:space:]]*(ifdef|ifndef)([[:space:]]|$)/ ||
                (line ~ /^[[:space:]]*#[[:space:]]*(if|elif)[[:space:]]+/ &&
                    one_expression(condition_expression(line)))
        }
        function unknown_condition(line) {
            return line ~ /^[[:space:]]*#[[:space:]]*(if|elif)([[:space:]]|$)/ &&
                !false_condition(line) && !true_condition(line)
        }
        function parent_active(    i) {
            for (i = 1; i < conditional_depth; i++)
                if (!conditional_active[i])
                    return 0
            return 1
        }
        function current_active(    i) {
            for (i = 1; i <= conditional_depth; i++)
                if (!conditional_active[i])
                    return 0
            return 1
        }
        function update_conditionals(line,    parent) {
            if (if_directive(line)) {
                conditional_depth++
                conditional_active[conditional_depth] = parent_active() && true_condition(line)
                conditional_taken[conditional_depth] = unknown_condition(line) || conditional_active[conditional_depth]
                return 1
            }
            if (elif_directive(line)) {
                parent = parent_active()
                conditional_active[conditional_depth] = parent && !conditional_taken[conditional_depth] && true_condition(line)
                conditional_taken[conditional_depth] = conditional_taken[conditional_depth] ||
                    unknown_condition(line) || conditional_active[conditional_depth]
                return 1
            }
            if (else_directive(line)) {
                parent = parent_active()
                conditional_active[conditional_depth] = parent && !conditional_taken[conditional_depth]
                conditional_taken[conditional_depth] = 1
                return 1
            }
            if (endif_directive(line)) {
                delete conditional_active[conditional_depth]
                delete conditional_taken[conditional_depth]
                conditional_depth--
                return 1
            }
            return 0
        }
        function strip_comments(line,    start, end, out) {
            out = ""
            while (length(line) > 0) {
                if (in_block_comment) {
                    end = index(line, "*/")
                    if (end == 0)
                        return out
                    line = substr(line, end + 2)
                    in_block_comment = 0
                    continue
                }
                start = index(line, "/*")
                if (start == 0) {
                    sub(/[[:space:]]*\/\/.*/, "", line)
                    return out line
                }
                out = out substr(line, 1, start - 1)
                line = substr(line, start + 2)
                in_block_comment = 1
            }
            return out
        }
AWK
)"

active_fixed_count_in_file() {
    local marker="$1"
    local file="$2"

    awk -v marker="$marker" "$ACTIVE_CODE_AWK"'
        {
            active = strip_comments($0)
            if (update_conditionals(active))
                next
            if (!current_active())
                next
            if (index(active, marker) > 0)
                count++
        }
        END { print count + 0 }
    ' "$file"
}

active_include_count() {
    local include_name="$1"
    local file="${2:-$TEST_FILE}"

    awk -v include_name="$include_name" "$ACTIVE_CODE_AWK"'
        function include_basename_matches(line, include_name,    target) {
            if (line !~ /^[[:space:]]*#[[:space:]]*include[[:space:]]*"/)
                return 0
            target = line
            sub(/^[[:space:]]*#[[:space:]]*include[[:space:]]*"/, "", target)
            sub(/".*$/, "", target)
            sub(/^.*\//, "", target)
            return target == include_name
        }
        {
            active = strip_comments($0)
            if (update_conditionals(active))
                next
            if (!current_active())
                next
            if (include_basename_matches(active, include_name))
                count++
        }
        END { print count + 0 }
    ' "$file"
}

active_run_test_count() {
    local marker="$1"

    awk -v marker="$marker" "$ACTIVE_CODE_AWK"'
        {
            active = strip_comments($0)
            if (update_conditionals(active))
                next
            if (!current_active())
                next
            if (active ~ /^[[:space:]]*RUN_TEST[[:space:]]*\(/ && index(active, marker) > 0)
                count++
        }
        END { print count + 0 }
    ' "$TEST_FILE"
}

active_run_test_line() {
    local marker="$1"

    awk -v marker="$marker" "$ACTIVE_CODE_AWK"'
        {
            active = strip_comments($0)
            if (update_conditionals(active))
                next
            if (!current_active())
                next
            if (active ~ /^[[:space:]]*RUN_TEST[[:space:]]*\(/ && index(active, marker) > 0) {
                print NR
                found = 1
                exit
            }
        }
        END {
            if (!found)
                print 0
        }
    ' "$TEST_FILE"
}

require_active_run_test_registration() {
    local marker="$1"
    local count

    count="$(active_run_test_count "$marker")"
    if [ "$count" -ne 1 ]; then
        fail "tests/test_ldlt_csc.c must retain proof-owner registration '$marker' exactly once as an active RUN_TEST line (found $count)"
    fi
}

require_increasing_run_test_order() {
    local marker
    local line
    local previous_line=0
    local solve_line

    for marker in "${RUN_TEST_MARKERS[@]}"; do
        line="$(active_run_test_line "$marker")"
        if [ "$line" -le "$previous_line" ]; then
            fail "tests/test_ldlt_csc.c selected RUN_TEST registrations changed order near '$marker'"
        fi
        previous_line="$line"
    done

    solve_line="$(active_run_test_line "$SOLVE_RUN_TEST_MARKER")"
    if [ -z "$solve_line" ]; then
        fail "tests/test_ldlt_csc.c must retain Day 9 solve registration '$SOLVE_RUN_TEST_MARKER'"
    fi
    if [ "$solve_line" -le "$previous_line" ]; then
        fail "tests/test_ldlt_csc.c selected RUN_TEST registrations must remain before Day 9 solve registration '$SOLVE_RUN_TEST_MARKER'"
    fi
}

require_test_ldlt_csc_rule_prerequisite() {
    local needle="$1"
    local message="$2"

    if ! awk -v needle="$needle" '
        BEGIN { done = 0; ok = 0 }
        /^\$\(BUILDDIR\)\/test_ldlt_csc:/ {
            done = 1
            rule = $0
            while (rule ~ /\\[[:space:]]*$/ && (getline next_line) > 0) {
                sub(/\\[[:space:]]*$/, " ", rule)
                rule = rule next_line
            }
            ok = index(rule, needle) > 0
            exit
        }
        END {
            if (!done || !ok)
                exit(1)
        }
    ' "$MAKEFILE"; then
        fail "$message"
    fi
}

require_makefile_target_command() {
    local target="$1"
    local command="$2"
    local message="$3"

    if ! awk -v target="$target" -v command="$command" '
        function active_recipe_line(line,    recipe) {
            if (line !~ /^[[:space:]]/)
                return 0
            recipe = line
            sub(/^[[:space:]]+/, "", recipe)
            while (recipe ~ /^[@+-]/) {
                sub(/^[@+-]/, "", recipe)
                sub(/^[[:space:]]+/, "", recipe)
            }
            return recipe !~ /^#/
        }
        BEGIN { in_target = 0; found_target = 0; found_command = 0 }
        /^[^[:space:]#][^:]*:/ {
            if (in_target)
                in_target = 0
        }
        index($0, target ":") == 1 {
            in_target = 1
            found_target = 1
            next
        }
        in_target && active_recipe_line($0) && index($0, command) > 0 {
            found_command = 1
            exit
        }
        END {
            if (!found_target || !found_command)
                exit(1)
        }
    ' "$MAKEFILE"; then
        fail "$message"
    fi
}

check_makefile_guard_wiring() {
    require_makefile_target_command "ldlt-csc-helper-guard" \
        "bash scripts/check_ldlt_csc_helper_guard.sh" \
        "Makefile ldlt-csc-helper-guard target must run scripts/check_ldlt_csc_helper_guard.sh"
    require_makefile_target_command "ldlt-csc-helper-guard" \
        "python3 tests/test_ldlt_csc_helper_guard.py" \
        "Makefile ldlt-csc-helper-guard target must run tests/test_ldlt_csc_helper_guard.py"
    require_makefile_target_command "ldlt-csc-helper-guard" \
        "python3 tests/test_ldlt_csc_native_parity_behavior.py" \
        "Makefile ldlt-csc-helper-guard target must run tests/test_ldlt_csc_native_parity_behavior.py"
    require_makefile_target_command "quality-review-compile" \
        "\$(MAKE) ldlt-csc-helper-guard" \
        "Makefile quality-review-compile target must run ldlt-csc-helper-guard"

    pass "Makefile guard wiring"
}

check_proof_owner_registration() {
    require_file "$TEST_FILE" "tests/test_ldlt_csc.c is missing"
    require_fixed '$(TESTDIR)/test_ldlt_csc.c' "$MAKEFILE" \
        "Makefile no longer registers test_ldlt_csc.c in TEST_SRCS"
    require_fixed 'add_sparse_test(test_ldlt_csc)' "$CMAKE_FILE" \
        "CMakeLists.txt no longer registers test_ldlt_csc"

    pass "proof-owner registration"
}

check_helper_headers() {
    local helper
    local helper_path
    local include_name
    local guard
    local count

    for helper in "${HELPERS[@]}"; do
        helper_path="$ROOT_DIR/$helper"
        include_name="$(basename "$helper")"
        guard="$(printf '%s' "$include_name" | tr '[:lower:].' '[:upper:]_')"

        require_file "$helper_path" "$helper is missing"
        require_fixed "#ifndef $guard" "$helper_path" "$helper is missing include guard $guard"
        require_fixed "#define $guard" "$helper_path" "$helper is missing include guard define $guard"
        count="$(active_include_count "$include_name")"
        if [ "$count" -ne 1 ]; then
            fail "tests/test_ldlt_csc.c must include $include_name exactly once as an active include (found $count)"
        fi
        count="$(fixed_occurrence_count "$include_name" "$MAKEFILE")"
        if [ "$count" -ne 1 ]; then
            fail "Makefile must list $include_name exactly once, only as a test_ldlt_csc prerequisite (found $count)"
        fi
        require_test_ldlt_csc_rule_prerequisite "\$(TESTDIR)/$include_name" \
            "Makefile test_ldlt_csc prerequisite rule must list $include_name"
    done

    pass "helper headers"
}

check_native_helper_translation_unit() {
    local include_name
    local file
    local rel_file
    local count

    include_name="$(basename "$NATIVE_HELPER")"

    while IFS= read -r file; do
        count="$(active_include_count "$include_name" "$file")"
        if [ "$file" = "$TEST_FILE" ]; then
            if [ "$count" -ne 1 ]; then
                fail "tests/test_ldlt_csc.c must be the single active translation-unit includer of $include_name"
            fi
        elif [ "$count" -ne 0 ]; then
            rel_file="${file#$ROOT_DIR/}"
            fail "$include_name must not be included by $rel_file"
        fi
    done < <(find "$ROOT_DIR" \
        \( -path "$ROOT_DIR/.git" -o -path "$ROOT_DIR/build" -o -path "$ROOT_DIR/docs/api" \) -prune -o \
        -type f \( -name '*.c' -o -name '*.h' \) -print | sort)

    pass "single translation-unit ownership"
}

check_header_only_registration() {
    local helper
    local include_name
    local stem

    require_file "$LIBRARY_MANIFEST" "build-metadata/library_sources.txt is missing"

    for helper in "${HELPERS[@]}"; do
        include_name="$(basename "$helper")"
        stem="${include_name%.h}"

        require_absent_fixed "$include_name" "$CMAKE_FILE" \
            "$include_name must remain header-only and not be named in CMake registration"
        require_absent_fixed "$helper" "$LIBRARY_MANIFEST" \
            "$helper must not be listed as a library source"
        require_absent_fixed "add_sparse_test($stem)" "$CMAKE_FILE" \
            "$stem must not become a separate CMake test without a new proof-owner decision"
    done

    pass "header-only registration"
}

check_selected_run_test_registrations() {
    local marker

    for marker in "${RUN_TEST_MARKERS[@]}"; do
        require_active_run_test_registration "$marker"
    done
    require_increasing_run_test_order

    pass "selected RUN_TEST registrations"
}

check_moved_definition_ownership() {
    local marker
    local helper
    local helper_path
    local count

    for marker in "${MOVED_DEFINITION_MARKERS[@]}"; do
        count="$(active_fixed_count_in_file "$marker" "$NATIVE_HELPER_PATH")"
        if [ "$count" -ne 1 ]; then
            fail "$NATIVE_HELPER must own moved selected-cluster definition '$marker' exactly once as active code (found $count)"
        fi

        count="$(active_fixed_count_in_file "$marker" "$TEST_FILE")"
        if [ "$count" -ne 0 ]; then
            fail "tests/test_ldlt_csc.c must not retain moved selected-cluster definition '$marker'"
        fi

        for helper in "${HELPERS[@]}"; do
            if [ "$helper" = "$NATIVE_HELPER" ]; then
                continue
            fi
            helper_path="$ROOT_DIR/$helper"
            count="$(active_fixed_count_in_file "$marker" "$helper_path")"
            if [ "$count" -ne 0 ]; then
                fail "$helper must not own moved selected-cluster definition '$marker'"
            fi
        done
    done

    pass "moved definition ownership"
}

check_proof_owner_registration
check_makefile_guard_wiring
check_helper_headers
check_native_helper_translation_unit
check_header_only_registration
check_selected_run_test_registrations
check_moved_definition_ownership

echo "ldlt-csc-helper-guard: passed"
