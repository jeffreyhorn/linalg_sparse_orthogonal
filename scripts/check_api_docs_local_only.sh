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
    local workflow_file
    local matches

    if [ ! -d "$workflows_dir" ]; then
        pass "no workflow directory for $label"
        return
    fi

    matches="$(
        for workflow_file in "$workflows_dir"/*.yml "$workflows_dir"/*.yaml; do
            [ -f "$workflow_file" ] || continue
            sed -E 's/[[:space:]]+#.*$//;/^[[:space:]]*#/d' "$workflow_file" |
                tr '\\' '/' |
                tr '[:upper:]' '[:lower:]' |
                grep -F -n "$needle" || true
        done
    )"
    if [ -n "$matches" ]; then
        printf '%s\n' "$matches" >&2
        fail "workflows must not reference $label while generated API HTML is strengthened local-only"
    fi

    pass "no workflow $label references"
}

require_workflows_do_not_match() {
    local pattern="$1"
    local label="$2"
    local workflows_dir="$ROOT_DIR/.github/workflows"
    local workflow_file
    local matches

    if [ ! -d "$workflows_dir" ]; then
        pass "no workflow directory for $label"
        return
    fi

    matches="$(
        for workflow_file in "$workflows_dir"/*.yml "$workflows_dir"/*.yaml; do
            [ -f "$workflow_file" ] || continue
            sed -E 's/[[:space:]]+#.*$//;/^[[:space:]]*#/d' "$workflow_file" |
                tr '\\' '/' |
                tr '[:upper:]' '[:lower:]' |
                grep -E -n "$pattern" || true
        done
    )"
    if [ -n "$matches" ]; then
        printf '%s\n' "$matches" >&2
        fail "workflows must not reference $label while generated API HTML is strengthened local-only"
    fi

    pass "no workflow $label references"
}

check_no_workflow_publication_semantics() {
    local workflows_dir="$ROOT_DIR/.github/workflows"
    local workflow_file
    local rel_path
    local publication_regex
    local command_publication_regex
    local generated_path_regex
    local broad_path_regex
    local broad_block_path_regex
    local broad_command_path_regex
    local docs_staging_command_regex
    local docs_archive_command_regex
    local dynamic_publication_path_regex
    local dynamic_command_publication_regex
    local broad_inline_path_regex
    local dynamic_inline_path_regex
    local stripped_text
    local normalized_text
    local dynamic_block_path_matches

    if [ ! -d "$workflows_dir" ]; then
        pass "no workflow directory for generated API publication semantics"
        return
    fi

    publication_regex="actions/upload-artifact|actions/upload-pages-artifact|actions/deploy-pages|actions/upload-release-asset|softprops/action-gh-release|github-pages|gh-pages|pages:|uses:[[:space:]]*['\"]?([^[:space:]'\"]+@|[.]/|docker://)"
    command_publication_regex="aws[[:space:]]+s3[[:space:]]+(sync|cp)|gsutil[[:space:]]+(-m[[:space:]]+)?(rsync|cp)|az[[:space:]]+storage[[:space:]]+blob[[:space:]]+upload|netlify[[:space:]]+deploy|vercel[[:space:]]+deploy|firebase[[:space:]]+deploy|wrangler[[:space:]]+pages[[:space:]]+deploy|surge[[:space:]]|gh[[:space:]]+release[[:space:]]+upload|rclone[[:space:]]+(copy|sync|move)|rsync[[:space:]].*:[^[:space:]]*|scp[[:space:]].*:[^[:space:]]*"
    generated_path_regex="docs/api(/|$)|docs/api/html"
    broad_path_regex='^[[:space:]]*["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*["'"'"']?(\.|[.]/|[.]/[*][*]|[*][*]([/][*])?|/|([.]/)?([^[:space:]"'"'"']+/)*([.][.]/)?docs($|[/.]|[*]|["'"'"'])[^[:space:]"'"'"']*|[$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?(/docs($|[/.]|[*]|["'"'"'])[^[:space:]"'"'"']*)?)["'"'"']?[[:space:]]*$'
    broad_block_path_regex='^[[:space:]]*-?[[:space:]]*["'"'"']?(\.|[.]/|[.]/[*][*]|[*][*]([/][*])?|/|([.]/)?([^[:space:]"'"'"']+/)*([.][.]/)?docs($|[/.]|[*]|["'"'"'])[^[:space:]"'"'"']*|[$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?(/docs($|[/.]|[*]|["'"'"'])[^[:space:]"'"'"']*)?)["'"'"']?[[:space:]]*$'
    broad_command_path_regex='(^|[[:space:]])["'"'"']?(([$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?)?([.]/)?docs($|[/.*[:space:]])|([.]/)?([*][*]([/][*])?|[.]([/][*][*])?)($|[[:space:]])|/($|[[:space:]]))'
    docs_staging_command_regex='(^|[[:space:]])(cp|mv|rsync)[[:space:]][^;&|]*["'"'"']?([.]/)?docs($|[/[:space:]"'"'"'])'
    docs_archive_command_regex='(^|[[:space:]])(tar|zip|7z|7za|7zr)[[:space:]][^;&|]*([[:space:]]|=)["'"'"']?([.]/)?docs($|[/[:space:]"'"'"'])'
    dynamic_publication_path_regex='^[[:space:]]*["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[^#]*([$][{][{]|[$][A-Za-z_][A-Za-z0-9_]*|[%][A-Za-z_][A-Za-z0-9_]*[%])'
    dynamic_command_publication_regex='(^|[[:space:]])["'"'"']?([$][A-Za-z_][A-Za-z0-9_]*|[$][{][A-Za-z_][A-Za-z0-9_]*[}]|[$][{][{][^}]+[}][}])'
    broad_inline_path_regex='(^|[,{][[:space:]]*)["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*["'"'"']?(\.|[.]/|[.]/[*][*]|[*][*]([/][*])?|/|([.]/)?([^[:space:],"'"'"'}]+/)*([.][.]/)?docs($|[/.]|[*]|["'"'"',}])[^[:space:],"'"'"'}]*|[$][{][{][[:space:]]*github[.]workspace[[:space:]]*[}][}]/?(/docs($|[/.]|[*]|["'"'"',}])[^[:space:],"'"'"'}]*)?)["'"'"']?([[:space:],}]|$)'
    dynamic_inline_path_regex='(^|[,{][[:space:]]*)["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[^,}]*([$][{][{]|[$][a-z_][a-z0-9_]*|[%][a-z_][a-z0-9_]*[%])'

    for workflow_file in "$workflows_dir"/*.yml "$workflows_dir"/*.yaml; do
        [ -f "$workflow_file" ] || continue
        rel_path="${workflow_file#$ROOT_DIR/}"
        stripped_text="$(sed -E 's/[[:space:]]+#.*$//;/^[[:space:]]*#/d' "$workflow_file")"
        normalized_text="$(printf '%s\n' "$stripped_text" | tr '\\' '/' | tr '[:upper:]' '[:lower:]')"
        dynamic_block_path_matches="$(
            printf '%s\n' "$normalized_text" |
                awk '
                    /^[[:space:]]*["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*[|>]/ {
                        in_path_block = 1
                        block_indent = match($0, /[^ ]/) - 1
                        next
                    }
                    in_path_block && /^[^[:space:]]/ {
                        in_path_block = 0
                    }
                    in_path_block && /^[[:space:]]*[^[:space:]]/ {
                        line_indent = match($0, /[^ ]/) - 1
                        if (line_indent <= block_indent) {
                            in_path_block = 0
                        } else if ($0 ~ /^[[:space:]]*["'"'"']?([$][{][{]|[$][a-z_][a-z0-9_]*|[%][a-z_][a-z0-9_]*[%])/) {
                            print
                        }
                    }
                '
        )"
        if printf '%s\n' "$normalized_text" | grep -Eq "$generated_path_regex" &&
            printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex"; then
            fail "$rel_path combines generated API output paths with publication, artifact, or Pages semantics while generated API HTML is local-only"
        fi
        if { printf '%s\n' "$normalized_text" | grep -Eq "$broad_path_regex" ||
            printf '%s\n' "$normalized_text" | grep -Eq "$broad_inline_path_regex"; } &&
            printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex"; then
            fail "$rel_path publishes docs or repository roots that can include local generated API HTML while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq '["'"'"']?(path|publish_dir|publish-dir|directory|folder|files|asset_path)["'"'"']?[[:space:]]*:[[:space:]]*[|>]' &&
            printf '%s\n' "$normalized_text" | grep -Eq "$broad_block_path_regex" &&
            printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex"; then
            fail "$rel_path publishes docs or repository roots that can include local generated API HTML while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$command_publication_regex" &&
            printf '%s\n' "$normalized_text" | grep -Eq "$broad_command_path_regex"; then
            fail "$rel_path publishes docs or repository roots that can include local generated API HTML while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex" &&
            printf '%s\n' "$normalized_text" | grep -Eq "$docs_staging_command_regex"; then
            fail "$rel_path stages docs for publication or artifact upload while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex" &&
            printf '%s\n' "$normalized_text" | grep -Eq "$docs_archive_command_regex"; then
            fail "$rel_path archives docs for publication or artifact upload while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$publication_regex" &&
            { printf '%s\n' "$normalized_text" | grep -Eq "$dynamic_publication_path_regex" ||
                printf '%s\n' "$normalized_text" | grep -Eq "$dynamic_inline_path_regex" ||
                [ -n "$dynamic_block_path_matches" ]; }; then
            fail "$rel_path uses dynamic publication paths while generated API HTML is local-only"
        fi
        if printf '%s\n' "$normalized_text" | grep -Eq "$command_publication_regex" &&
            printf '%s\n' "$normalized_text" | grep -Eq "$dynamic_command_publication_regex"; then
            fail "$rel_path uses dynamic publication paths while generated API HTML is local-only"
        fi
    done

    pass "no workflow generated API publication semantics"
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

    staged="$(git -C "$ROOT_DIR" diff --cached --name-only -- docs/api)"
    require_empty \
        "no staged generated API files" \
        "generated API files under docs/api/ are staged; unstage them unless a future publication decision selects committed output" \
        "$staged"

    tracked="$(git -C "$ROOT_DIR" ls-files docs/api)"
    require_empty \
        "no tracked generated API files" \
        "generated API files under docs/api/ are tracked; local-only generated HTML must not be source-controlled" \
        "$tracked"

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
    check_no_workflow_publication_semantics
    require_workflows_do_not_match \
        '(^|[^[:alnum:]_.-])([.]/|/)?docs/api([^[:alnum:]_./-]|$)' \
        "generated API output root"
    require_workflows_do_not_reference "docs/api/html" "generated API HTML output path"
    require_workflows_do_not_reference "docs/api/" "generated API output tree"
}

check_tracked_and_staged_absence
check_ignore_rules
check_doxyfile_contract
check_product_status_wording
check_no_workflow_publication_path

echo "api-docs-local-only: passed"
