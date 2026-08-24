#!/usr/bin/env bash
# check_api_docs_local_only.sh - generated API HTML local-only guard.
#
# Proves that Doxygen HTML under docs/api/ remains ignored local generated
# output unless a future publication decision explicitly selects committed
# generated API HTML.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

fail() {
    echo "api-docs-local-only: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "api-docs-local-only: $1 ok"
}

require_ignored() {
    local path="$1"

    if ! git -C "$ROOT_DIR" check-ignore -q "$path"; then
        fail "$path is not ignored; generated API HTML must remain local-only unless a future publication decision selects committed output"
    fi

    pass "$path ignore rule"
}

require_empty() {
    local label="$1"
    local message="$2"
    local output="$3"

    if [ -n "$output" ]; then
        printf '%s\n' "$output" >&2
        fail "$message"
    fi

    pass "$label"
}

require_file_contains() {
    local path="$1"
    local needle="$2"
    local label="$3"
    local full_path="$ROOT_DIR/$path"

    if [ ! -f "$full_path" ]; then
        fail "$path is missing; cannot verify strengthened local-only generated API HTML product decision wording"
    fi

    if ! grep -Fq "$needle" "$full_path"; then
        fail "$path must state $label for the strengthened local-only generated API HTML product decision"
    fi

    pass "$path $label wording"
}

require_doxyfile_setting() {
    local key="$1"
    local value="$2"
    local label="$3"
    local doxyfile="$ROOT_DIR/Doxyfile"

    if [ ! -f "$doxyfile" ]; then
        fail "Doxyfile is missing; cannot verify generated API HTML local-only output contract"
    fi

    if ! grep -Eq "^[[:space:]]*$key[[:space:]]*=[[:space:]]*$value[[:space:]]*$" "$doxyfile"; then
        fail "Doxyfile must keep $key = $value for $label"
    fi

    pass "Doxyfile $key local-only contract"
}

require_workflows_do_not_reference() {
    local needle="$1"
    local label="$2"
    local workflows_dir="$ROOT_DIR/.github/workflows"
    local matches

    if [ ! -d "$workflows_dir" ]; then
        pass "no workflow directory for $label"
        return
    fi

    matches="$(grep -R -F -n "$needle" "$workflows_dir" || true)"
    if [ -n "$matches" ]; then
        printf '%s\n' "$matches" >&2
        fail "workflows must not reference $label while generated API HTML is strengthened local-only"
    fi

    pass "no workflow $label references"
}

check_ignore_rules() {
    require_ignored "docs/api"
    require_ignored "docs/api/html"
    require_ignored "docs/api/html/index.html"
}

check_tracked_and_staged_absence() {
    local tracked
    local staged
    local visible_untracked

    tracked="$(git -C "$ROOT_DIR" ls-files docs/api)"
    require_empty \
        "no tracked generated API files" \
        "generated API files under docs/api/ are tracked; local-only generated HTML must not be source-controlled" \
        "$tracked"

    staged="$(git -C "$ROOT_DIR" diff --cached --name-only -- docs/api)"
    require_empty \
        "no staged generated API files" \
        "generated API files under docs/api/ are staged; unstage them unless a future publication decision selects committed output" \
        "$staged"

    visible_untracked="$(git -C "$ROOT_DIR" ls-files --others --exclude-standard docs/api)"
    require_empty \
        "no non-ignored generated API files" \
        "generated API files under docs/api/ are visible as non-ignored untracked files; keep local generated output ignored" \
        "$visible_untracked"
}

check_doxyfile_contract() {
    require_doxyfile_setting "INPUT" "include/" "public-header generated API input"
    require_doxyfile_setting "FILE_PATTERNS" "\\*.h" "public-header generated API input"
    require_doxyfile_setting "RECURSIVE" "NO" "top-level public-header scope"
    require_doxyfile_setting "OUTPUT_DIRECTORY" "docs/api" "ignored generated API output"
    require_doxyfile_setting "GENERATE_HTML" "YES" "local generated API HTML"
    require_doxyfile_setting "HTML_OUTPUT" "html" "ignored generated API HTML output"
}

check_product_status_wording() {
    require_file_contains \
        "README.md" \
        "selected local Doxygen freshness plus local-only staging guard" \
        "local-only freshness"

    require_file_contains \
        "docs/api_reference.md" \
        "The generated HTML tree is local-only generated output." \
        "local-only generated output"

    require_file_contains \
        "docs/api_reference.md" \
        "is not a hosted or source-controlled publication surface." \
        "not hosted or source-controlled"

    require_file_contains \
        "docs/maintainer_guide.md" \
        "The maintained Sprint 179 product decision keeps this tree" \
        "Sprint 179 product decision"

    require_file_contains \
        "docs/maintainer_guide.md" \
        "hosted, artifact-published, or release evidence." \
        "not hosted, artifact-published, or release evidence"
}

check_no_workflow_publication_path() {
    require_workflows_do_not_reference "docs/api/html" "generated API HTML output path"
    require_workflows_do_not_reference "docs/api/" "generated API output tree"
}

check_ignore_rules
check_doxyfile_contract
check_tracked_and_staged_absence
check_product_status_wording
check_no_workflow_publication_path

echo "api-docs-local-only: passed"
