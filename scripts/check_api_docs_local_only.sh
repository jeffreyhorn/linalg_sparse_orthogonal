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

check_ignore_rules
check_tracked_and_staged_absence

echo "api-docs-local-only: passed"
