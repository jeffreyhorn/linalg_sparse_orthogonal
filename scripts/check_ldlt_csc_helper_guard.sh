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
BEHAVIOR_FILE="$ROOT_DIR/tests/test_ldlt_csc_native_parity_behavior.py"
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

makefile_active_occurrence_count() {
    local needle="$1"
    local file="$2"

    awk -v needle="$needle" '
        function strip_make_comment(line) {
            sub(/[[:space:]]*#.*/, "", line)
            return line
        }
        {
            line = strip_make_comment($0)
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
        function macro_name(line,    name) {
            name = line
            sub(/^[[:space:]]*#[[:space:]]*(ifdef|ifndef)[[:space:]]+/, "", name)
            sub(/[[:space:]].*$/, "", name)
            return name
        }
        function include_guard_condition(line) {
            return include_guard != "" &&
                line ~ /^[[:space:]]*#[[:space:]]*ifndef[[:space:]]+/ &&
                macro_name(line) == include_guard
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
            return include_guard_condition(line) ||
                (line ~ /^[[:space:]]*#[[:space:]]*(if|elif)[[:space:]]+/ &&
                    one_expression(condition_expression(line)))
        }
        function unknown_condition(line) {
            return line ~ /^[[:space:]]*#[[:space:]]*(if|ifdef|ifndef|elif)([[:space:]]|$)/ &&
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
                if (fail_closed_unknown && unknown_condition(line)) {
                    conditional_active[conditional_depth] = parent_active()
                    conditional_taken[conditional_depth] = 0
                } else {
                    conditional_active[conditional_depth] = parent_active() && true_condition(line)
                    conditional_taken[conditional_depth] = unknown_condition(line) || conditional_active[conditional_depth]
                }
                return 1
            }
            if (elif_directive(line)) {
                parent = parent_active()
                if (fail_closed_unknown && unknown_condition(line)) {
                    conditional_active[conditional_depth] = parent && !conditional_taken[conditional_depth]
                } else {
                    conditional_active[conditional_depth] = parent && !conditional_taken[conditional_depth] && true_condition(line)
                    conditional_taken[conditional_depth] = conditional_taken[conditional_depth] ||
                        unknown_condition(line) || conditional_active[conditional_depth]
                }
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

include_guard_for_file() {
    local file="$1"

    awk '
        /^[[:space:]]*#[[:space:]]*ifndef[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/ {
            guard = $0
            sub(/^[[:space:]]*#[[:space:]]*ifndef[[:space:]]+/, "", guard)
            sub(/[[:space:]].*$/, "", guard)
            next
        }
        guard != "" {
            pattern = "^[[:space:]]*#[[:space:]]*define[[:space:]]+" guard "([[:space:]]|$)"
            if ($0 ~ pattern) {
                print guard
                exit
            }
        }
        NR > 20 {
            exit
        }
    ' "$file"
}

active_fixed_count_in_file() {
    local marker="$1"
    local file="$2"
    local fail_closed_unknown="${3:-0}"
    local include_guard

    include_guard="$(include_guard_for_file "$file")"

    awk -v marker="$marker" -v include_guard="$include_guard" \
        -v fail_closed_unknown="$fail_closed_unknown" "$ACTIVE_CODE_AWK"'
        {
            active = strip_comments($0)
            if (update_conditionals(active))
                next
            if (!current_active())
                next
            while ((pos = index(active, marker)) > 0) {
                count++
                active = substr(active, pos + length(marker))
            }
        }
        END { print count + 0 }
    ' "$file"
}

active_include_count() {
    local include_name="$1"
    local file="${2:-$TEST_FILE}"
    local fail_closed_unknown="${3:-0}"
    local include_guard

    include_guard="$(include_guard_for_file "$file")"

    awk -v include_name="$include_name" -v include_guard="$include_guard" \
        -v fail_closed_unknown="$fail_closed_unknown" "$ACTIVE_CODE_AWK"'
        function include_basename_matches(line, include_name,    target) {
            if (line !~ /^[[:space:]]*#[[:space:]]*include[[:space:]]*["<]/)
                return 0
            target = line
            sub(/^[[:space:]]*#[[:space:]]*include[[:space:]]*["<]/, "", target)
            sub(/[">].*$/, "", target)
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
    local fail_closed_unknown="${2:-0}"

    awk -v marker="$marker" -v include_guard="" \
        -v fail_closed_unknown="$fail_closed_unknown" "$ACTIVE_CODE_AWK"'
        {
            active = strip_comments($0)
            if (update_conditionals(active))
                next
            if (!current_active())
                next
            if (active ~ /^[[:space:]]*RUN_TEST[[:space:]]*\(/) {
                line = active
                while ((pos = index(line, marker)) > 0) {
                    count++
                    line = substr(line, pos + length(marker))
                }
            }
        }
        END { print count + 0 }
    ' "$TEST_FILE"
}

active_run_test_line() {
    local marker="$1"

    awk -v marker="$marker" -v include_guard="" "$ACTIVE_CODE_AWK"'
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
    local possible_count

    count="$(active_run_test_count "$marker")"
    possible_count="$(active_run_test_count "$marker" 1)"
    if [ "$count" -ne 1 ] || [ "$possible_count" -ne 1 ]; then
        fail "tests/test_ldlt_csc.c must retain proof-owner registration '$marker' exactly once as an active RUN_TEST line and no additional possible-active registrations (active $count, possible $possible_count)"
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
    if [ "$solve_line" -eq 0 ]; then
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
        function strip_make_comment(line) {
            sub(/[[:space:]]*#.*/, "", line)
            return line
        }
        function has_prerequisite_token(rule, needle,    field_count, i, fields) {
            field_count = split(rule, fields, /[[:space:]]+/)
            for (i = 1; i <= field_count; i++)
                if (fields[i] == needle)
                    return 1
            return 0
        }
        BEGIN { done = 0; ok = 0 }
        /^\$\(BUILDDIR\)\/test_ldlt_csc:/ {
            done = 1
            rule = strip_make_comment($0)
            while (rule ~ /\\[[:space:]]*$/ && (getline next_line) > 0) {
                sub(/\\[[:space:]]*$/, " ", rule)
                rule = rule strip_make_comment(next_line)
            }
            ok = has_prerequisite_token(rule, needle)
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
        function parsed_recipe_command(line,    recipe) {
            if (line !~ /^[[:space:]]/)
                return ""
            recipe = line
            sub(/^[[:space:]]+/, "", recipe)
            while (recipe ~ /^[@+-]/) {
                sub(/^[@+-]/, "", recipe)
                sub(/^[[:space:]]+/, "", recipe)
            }
            sub(/[[:space:]]+#.*$/, "", recipe)
            sub(/[[:space:]]+$/, "", recipe)
            if (recipe ~ /^#/)
                return ""
            return recipe
        }
        function command_matches(line, command,    recipe, suffix) {
            recipe = parsed_recipe_command(line)
            if (recipe == command)
                return 1
            if (index(recipe, command) != 1)
                return 0
            suffix = substr(recipe, length(command) + 1, 1)
            return suffix ~ /[[:space:]]/
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
        in_target && command_matches($0, command) {
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
    require_makefile_target_command "quality-review-full" \
        "\$(MAKE) quality-review-compile" \
        "Makefile quality-review-full target must run quality-review-compile"

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
    local possible_count

    for helper in "${HELPERS[@]}"; do
        helper_path="$ROOT_DIR/$helper"
        include_name="$(basename "$helper")"
        guard="$(printf '%s' "$include_name" | tr '[:lower:].' '[:upper:]_')"

        require_file "$helper_path" "$helper is missing"
        require_fixed "#ifndef $guard" "$helper_path" "$helper is missing include guard $guard"
        require_fixed "#define $guard" "$helper_path" "$helper is missing include guard define $guard"
        count="$(active_include_count "$include_name")"
        possible_count="$(active_include_count "$include_name" "$TEST_FILE" 1)"
        if [ "$count" -ne 1 ] || [ "$possible_count" -ne 1 ]; then
            fail "tests/test_ldlt_csc.c must include $include_name exactly once as an active include and no additional possible-active includes (active $count, possible $possible_count)"
        fi
        count="$(makefile_active_occurrence_count "$include_name" "$MAKEFILE")"
        if [ "$count" -ne 1 ]; then
            fail "Makefile must list exact prerequisite token \$(TESTDIR)/$include_name exactly once, only as a test_ldlt_csc prerequisite (found $count)"
        fi
        require_test_ldlt_csc_rule_prerequisite "\$(TESTDIR)/$include_name" \
            "Makefile test_ldlt_csc prerequisite rule must list exact prerequisite token \$(TESTDIR)/$include_name"
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
        elif [ "$(active_include_count "$include_name" "$file" 1)" -ne 0 ]; then
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
        require_absent_fixed "$include_name" "$LIBRARY_MANIFEST" \
            "$include_name must not be listed as a library source"
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

check_behavior_regression_contract() {
    local selected_tests=()
    local marker

    require_file "$BEHAVIOR_FILE" "tests/test_ldlt_csc_native_parity_behavior.py is missing"
    for marker in "${RUN_TEST_MARKERS[@]}"; do
        marker="${marker#RUN_TEST(}"
        marker="${marker%);}"
        selected_tests+=("$marker")
    done

    python3 - "$BEHAVIOR_FILE" "${selected_tests[@]}" <<'PY'
import ast
import sys
from pathlib import Path

behavior_path = Path(sys.argv[1])
expected_selected = list(sys.argv[2:])
expected_summary = {
    "Tests run": 100,
    "Tests failed": 0,
    "Tests skipped": 0,
    "Assertions": 3556,
}


def fail(message: str) -> None:
    raise SystemExit(message)


tree = ast.parse(behavior_path.read_text(encoding="utf-8"), filename=str(behavior_path))
assignments: dict[str, ast.AST] = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value

try:
    selected_tests = ast.literal_eval(assignments["SELECTED_TESTS"])
except (KeyError, ValueError, SyntaxError):
    fail("behavior suite must define literal SELECTED_TESTS")
if selected_tests != expected_selected:
    fail("behavior suite must pin selected native parity tests in SELECTED_TESTS")

try:
    baseline_summary = ast.literal_eval(assignments["BASELINE_SUMMARY"])
except (KeyError, ValueError, SyntaxError):
    fail("behavior suite must define literal BASELINE_SUMMARY")
for key, value in expected_summary.items():
    if baseline_summary.get(key) != value:
        fail(f"behavior suite must preserve {key!r} summary value {value}")
if baseline_summary != expected_summary:
    fail("behavior suite must preserve exact test_ldlt_csc summary contract")


def is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def is_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and is_name(node.func, name)


has_selected_check = any(
    is_call(node, "assert_selected_tests_passed_in_order")
    and len(node.args) == 1
    and is_name(node.args[0], "output")
    for node in ast.walk(tree)
)
if not has_selected_check:
    fail("behavior suite must check selected native parity pass-marker order")

has_summary_compare = any(
    isinstance(node, ast.Compare)
    and is_name(node.left, "summary")
    and len(node.ops) == 1
    and isinstance(node.ops[0], ast.NotEq)
    and len(node.comparators) == 1
    and is_name(node.comparators[0], "BASELINE_SUMMARY")
    for node in ast.walk(tree)
)
if not has_summary_compare:
    fail("behavior suite must compare the full baseline summary")

has_all_passed_check = any(
    isinstance(node, ast.Compare)
    and isinstance(node.left, ast.Constant)
    and node.left.value == "ALL TESTS PASSED"
    and len(node.ops) == 1
    and isinstance(node.ops[0], ast.NotIn)
    and len(node.comparators) == 1
    and is_name(node.comparators[0], "output")
    for node in ast.walk(tree)
)
if not has_all_passed_check:
    fail("behavior suite must preserve all-tests-passed diagnostic check")

has_output_excerpt = any(
    is_call(node, "output_excerpt")
    and len(node.args) >= 1
    and is_name(node.args[0], "output")
    for node in ast.walk(tree)
)
if not has_output_excerpt:
    fail("behavior suite must preserve command-output diagnostics on failure")
PY

    pass "behavior regression contract"
}

check_moved_definition_ownership() {
    local marker
    local file
    local rel_file
    local count
    local possible_count

    for marker in "${MOVED_DEFINITION_MARKERS[@]}"; do
        count="$(active_fixed_count_in_file "$marker" "$NATIVE_HELPER_PATH")"
        possible_count="$(active_fixed_count_in_file "$marker" "$NATIVE_HELPER_PATH" 1)"
        if [ "$count" -ne 1 ] || [ "$possible_count" -ne 1 ]; then
            fail "$NATIVE_HELPER must own moved selected-cluster definition '$marker' exactly once as active code and no additional possible-active definitions (active $count, possible $possible_count)"
        fi

        while IFS= read -r file; do
            if [ "$file" = "$NATIVE_HELPER_PATH" ]; then
                continue
            fi
            count="$(active_fixed_count_in_file "$marker" "$file" 1)"
            if [ "$count" -ne 0 ]; then
                rel_file="${file#$ROOT_DIR/}"
                if [ "$file" = "$TEST_FILE" ]; then
                    fail "tests/test_ldlt_csc.c must not retain moved selected-cluster definition '$marker'"
                fi
                fail "$rel_file must not own moved selected-cluster definition '$marker'"
            fi
        done < <(find "$ROOT_DIR" \
            \( -path "$ROOT_DIR/.git" -o -path "$ROOT_DIR/build" -o -path "$ROOT_DIR/docs/api" \) -prune -o \
            -type f \( -name '*.c' -o -name '*.h' \) -print | sort)
    done

    pass "moved definition ownership"
}

check_proof_owner_registration
check_makefile_guard_wiring
check_helper_headers
check_native_helper_translation_unit
check_header_only_registration
check_selected_run_test_registrations
check_behavior_regression_contract
check_moved_definition_ownership

echo "ldlt-csc-helper-guard: passed"
