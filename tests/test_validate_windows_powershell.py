#!/usr/bin/env python3
"""Guard Windows PowerShell validation ownership checks."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_windows_powershell as validator  # noqa: E402


def read_workflow() -> str:
    return validator.WINDOWS_WORKFLOW.read_text(encoding="utf-8")


def assert_raises_with(fn, expected: str) -> None:
    try:
        fn()
    except validator.ValidationError as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}") from exc
        return
    raise AssertionError(f"expected validation failure containing {expected!r}")


def selected_steps_from(text: str) -> list[validator.Step]:
    rows = validator.selected_report_targets()
    steps = validator.validate_workflow_structure(
        text,
        validator.selected_report_freshness_tokens(rows),
    )
    return validator.validate_required_steps(steps)


def manifest_rows() -> list[dict[str, str]]:
    return validator.selected_report_targets()


def selected_steps() -> list[validator.Step]:
    return selected_steps_from(read_workflow())


def write_fake_pwsh(directory: Path, body: str) -> Path:
    fake = directory / "pwsh"
    fake.write_text("#!/usr/bin/env sh\n" + body, encoding="utf-8")
    fake.chmod(0o755)
    return fake


def run_with_path(argv: list[str], path: str) -> int:
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = path
    try:
        return validator.main(argv)
    finally:
        os.environ["PATH"] = old_path


def test_current_windows_workflow_structural_validation() -> None:
    validator.validate_hosted_validation_wiring(read_workflow())
    assert len(selected_steps()) == len(validator.STEP_REQUIREMENTS)


def test_shell_drift_fails_clearly() -> None:
    drifted = read_workflow().replace("        shell: pwsh", "        shell: cmd", 1)
    steps = validator.validate_workflow_structure(drifted)
    assert_raises_with(
        lambda: validator.validate_required_steps(steps),
        "must declare shell: pwsh",
    )


def test_command_anchor_drift_fails_clearly() -> None:
    drifted = read_workflow().replace("cmake -S . -B build", "cmake --preset windows", 1)
    steps = validator.validate_workflow_structure(drifted)
    assert_raises_with(
        lambda: validator.validate_required_steps(steps),
        "missing token 'cmake -S . -B build'",
    )


def test_unowned_powershell_step_fails_clearly() -> None:
    marker = "      - name: Run enforced reviewed CMake configure path"
    injected = (
        "      - name: Unowned PowerShell report-adjacent drift\n"
        "        run: Write-Host \"drift\"\n"
        "        shell: pwsh\n\n"
    )
    drifted = read_workflow().replace(marker, injected + marker, 1)
    steps = validator.validate_workflow_structure(drifted)
    assert_raises_with(
        lambda: validator.validate_required_steps(steps),
        "windows workflow has unowned PowerShell steps",
    )


def test_forbidden_windows_report_freshness_command_fails_clearly() -> None:
    drifted = read_workflow() + "\n# drift\nrun: make report-index-comparison-freshness\n"
    assert_raises_with(
        lambda: validator.validate_workflow_structure(
            drifted,
            validator.selected_report_freshness_tokens(manifest_rows()),
        ),
        "windows workflow must not run or upload selected report freshness",
    )


def test_extra_windows_upload_artifact_fails_outside_selected_lane() -> None:
    drifted = read_workflow() + "\n      - uses: actions/upload-artifact@v4\n"
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "must not publish hosted artifacts outside the selected Cholesky freshness lane",
    )


def test_selected_cholesky_lane_missing_target_fails_clearly() -> None:
    drifted = read_workflow().replace("--selected-target cholesky-spd-tridiag-5", "", 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "missing selected Cholesky token",
    )


def test_selected_cholesky_lane_missing_timeout_fails_clearly() -> None:
    drifted = read_workflow().replace("    timeout-minutes: 20\n", "", 1)
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "must declare timeout-minutes: 20",
    )


def test_selected_cholesky_lane_broad_upload_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "            build/comparison/cholesky_spd_tridiag_5/project_observations.tsv",
        "            build/comparison/**",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "must not use broad comparison artifact paths",
    )


def test_selected_cholesky_lane_missing_required_upload_fails_clearly() -> None:
    drifted = read_workflow().replace(
        "            build/comparison/cholesky_spd_tridiag_5/manifest.tsv\n",
        "",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_workflow_structure(drifted),
        "missing upload path",
    )


def test_manifest_derived_artifact_name_is_forbidden_on_windows() -> None:
    artifact = manifest_rows()[0]["workflow_artifact"]
    drifted = read_workflow() + f"\n# drift\nname: {artifact}\n"
    assert_raises_with(
        lambda: validator.validate_workflow_structure(
            drifted,
            validator.selected_report_freshness_tokens(manifest_rows()),
        ),
        f"selected report freshness {artifact!r}",
    )


def test_claim_boundaries_validate_current_docs() -> None:
    validator.validate_claim_boundaries()


def test_claim_boundary_missing_marker_fails_clearly() -> None:
    path = validator.REPO_ROOT / "README.md"
    text = path.read_text(encoding="utf-8").replace(
        "bounded Windows selected Cholesky comparison freshness workflow",
        "Windows report freshness follows the hosted validation lane",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "missing Windows/PowerShell non-claim marker",
    )


def test_claim_boundary_promotion_wording_fails_clearly() -> None:
    path = validator.REPO_ROOT / "docs" / "maintainer_guide.md"
    text = path.read_text(encoding="utf-8") + (
        "\nPowerShell validation proves Windows report freshness.\n"
    )
    assert_raises_with(
        lambda: validator.validate_claim_boundaries({path: text}),
        "unsupported Windows/PowerShell claim",
    )


def test_hosted_validation_wiring_requires_fail_closed_command() -> None:
    drifted = read_workflow().replace(" --require-pwsh", "", 1)
    assert_raises_with(
        lambda: validator.validate_hosted_validation_wiring(drifted),
        "must run",
    )


def test_hosted_validation_wiring_requires_windows_runner() -> None:
    drifted = read_workflow().replace(
        "  powershell-validation:\n"
        "    name: Windows PowerShell validation ownership\n"
        "    runs-on: windows-2022",
        "  powershell-validation:\n"
        "    name: Windows PowerShell validation ownership\n"
        "    runs-on: ubuntu-latest",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_hosted_validation_wiring(drifted),
        "must run on windows-2022",
    )


def test_hosted_validation_wiring_does_not_use_pwsh_shell() -> None:
    drifted = read_workflow().replace(
        "        run: python scripts/validate_windows_powershell.py --require-pwsh\n"
        "        shell: cmd",
        "        run: python scripts/validate_windows_powershell.py --require-pwsh\n"
        "        shell: pwsh",
        1,
    )
    assert_raises_with(
        lambda: validator.validate_hosted_validation_wiring(drifted),
        "must declare shell: cmd",
    )


def test_hosted_validation_pwsh_shell_is_unowned() -> None:
    drifted = read_workflow().replace(
        "        run: python scripts/validate_windows_powershell.py --require-pwsh\n"
        "        shell: cmd",
        "        run: python scripts/validate_windows_powershell.py --require-pwsh\n"
        "        shell: pwsh",
        1,
    )
    steps = validator.validate_workflow_structure(drifted)
    assert_raises_with(
        lambda: validator.validate_required_steps(steps),
        "windows workflow has unowned PowerShell steps",
    )


def test_selected_report_manifest_references_validate() -> None:
    validator.validate_selected_report_references(manifest_rows())


def test_missing_manifest_workflow_file_fails_clearly() -> None:
    rows = manifest_rows()
    rows[0] = dict(rows[0])
    rows[0]["workflow_file"] = ".github/workflows/missing-windows.yml"
    assert_raises_with(
        lambda: validator.validate_selected_report_references(rows),
        "references missing workflow_file",
    )


def test_manifest_windows_deferral_validation() -> None:
    validator.validate_manifest_windows_deferral(manifest_rows())


def test_deferral_record_validation() -> None:
    validator.validate_deferral_record()


def test_parse_with_fake_pwsh_accepts_selected_snippets() -> None:
    fake_body = """
test "$1" = "-NoProfile" || exit 11
test "$2" = "-NonInteractive" || exit 12
test "$3" = "-Command" || exit 13
test -n "$SPARSE_PWSH_SNIPPET" || exit 14
test -f "$SPARSE_PWSH_SNIPPET" || exit 15
grep -Eq 'cmake|ctest|sparse_lu_ortho.lib' "$SPARSE_PWSH_SNIPPET" || exit 16
printf '%s\\n' "$SPARSE_PWSH_SNIPPET" >> "$SPARSE_FAKE_PWSH_LOG"
"""
    with tempfile.TemporaryDirectory(prefix="sparse-fake-pwsh-") as tmp:
        tmpdir = Path(tmp)
        fake = write_fake_pwsh(tmpdir, fake_body)
        log = tmpdir / "pwsh.log"
        old_log = os.environ.get("SPARSE_FAKE_PWSH_LOG")
        os.environ["SPARSE_FAKE_PWSH_LOG"] = str(log)
        try:
            validator.parse_with_pwsh(str(fake), selected_steps())
        finally:
            if old_log is None:
                os.environ.pop("SPARSE_FAKE_PWSH_LOG", None)
            else:
                os.environ["SPARSE_FAKE_PWSH_LOG"] = old_log
        assert len(log.read_text(encoding="utf-8").splitlines()) == len(
            validator.STEP_REQUIREMENTS
        )


def test_parse_with_fake_pwsh_failure_is_actionable() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-fake-pwsh-") as tmp:
        fake = write_fake_pwsh(Path(tmp), "printf 'parse bad\\n' >&2\nexit 42\n")
        assert_raises_with(
            lambda: validator.parse_with_pwsh(str(fake), selected_steps()),
            "PowerShell parse failed",
        )


def test_main_with_fake_pwsh_returns_pass() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-fake-pwsh-") as tmp:
        tmpdir = Path(tmp)
        write_fake_pwsh(
            tmpdir,
            """
test -n "$SPARSE_PWSH_SNIPPET" || exit 14
test -f "$SPARSE_PWSH_SNIPPET" || exit 15
exit 0
""",
        )
        fake_pwsh_path = str(tmpdir) + os.pathsep + os.environ.get("PATH", "")
        assert run_with_path([], fake_pwsh_path) == 0
        assert run_with_path(["--require-pwsh"], fake_pwsh_path) == 0


def test_local_missing_pwsh_returns_unavailable() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-no-pwsh-") as tmp:
        assert run_with_path([], tmp) == 2


def test_require_pwsh_fails_closed_when_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-no-pwsh-") as tmp:
        assert run_with_path(["--require-pwsh"], tmp) == 1


def test_unavailable_output_keeps_non_pass_evidence_wording() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse-no-pwsh-") as tmp:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = run_with_path([], tmp)
    assert rc == 2
    output = stderr.getvalue()
    assert "UNAVAILABLE: pwsh not found; structural checks passed" in output
    assert "local unavailable PowerShell is not pass evidence" in output


if __name__ == "__main__":
    test_current_windows_workflow_structural_validation()
    test_shell_drift_fails_clearly()
    test_command_anchor_drift_fails_clearly()
    test_unowned_powershell_step_fails_clearly()
    test_forbidden_windows_report_freshness_command_fails_clearly()
    test_extra_windows_upload_artifact_fails_outside_selected_lane()
    test_selected_cholesky_lane_missing_target_fails_clearly()
    test_selected_cholesky_lane_missing_timeout_fails_clearly()
    test_selected_cholesky_lane_broad_upload_fails_clearly()
    test_selected_cholesky_lane_missing_required_upload_fails_clearly()
    test_manifest_derived_artifact_name_is_forbidden_on_windows()
    test_claim_boundaries_validate_current_docs()
    test_claim_boundary_missing_marker_fails_clearly()
    test_claim_boundary_promotion_wording_fails_clearly()
    test_hosted_validation_wiring_requires_fail_closed_command()
    test_hosted_validation_wiring_requires_windows_runner()
    test_hosted_validation_wiring_does_not_use_pwsh_shell()
    test_hosted_validation_pwsh_shell_is_unowned()
    test_selected_report_manifest_references_validate()
    test_missing_manifest_workflow_file_fails_clearly()
    test_manifest_windows_deferral_validation()
    test_deferral_record_validation()
    test_parse_with_fake_pwsh_accepts_selected_snippets()
    test_parse_with_fake_pwsh_failure_is_actionable()
    test_main_with_fake_pwsh_returns_pass()
    test_local_missing_pwsh_returns_unavailable()
    test_require_pwsh_fails_closed_when_missing()
    test_unavailable_output_keeps_non_pass_evidence_wording()
    print("test-validate-windows-powershell: ok")
