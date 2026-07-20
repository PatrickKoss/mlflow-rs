"""T22.1 manifest-generated semantic differential corpus.

The required profile replays committed Python-reference goldens through the
native engines and runs the lightweight live Python/Rust oracles. The refresh
profile additionally regenerates the dependency-heavy third-party golden with
the exact pinned packages; it is intended for nightly/manual CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUST_ROOT = ROOT / "rust"
REPORT_DIR = RUST_ROOT / "compliance" / "report"
INVENTORY_DIR = RUST_ROOT / "genai-inventory"
FIXTURE_DIR = RUST_ROOT / "crates" / "mlflow-genai" / "tests" / "fixtures"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    cwd: Path


@dataclass
class CommandResult:
    name: str
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class SemanticCase:
    section: str
    name: str
    status: str
    oracle: str
    diff_count: int


def _load(path: Path):
    return json.loads(path.read_text())


def _run(spec: CommandSpec) -> CommandResult:
    completed = subprocess.run(
        spec.argv,
        cwd=spec.cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(
        name=spec.name,
        argv=list(spec.argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _case_names() -> dict[str, list[str]]:
    scorers = _load(INVENTORY_DIR / "scorers.json")
    providers = _load(INVENTORY_DIR / "provider_manifest.json")
    algorithms = _load(INVENTORY_DIR / "algorithms.json")
    discovery = _load(FIXTURE_DIR / "issue_discovery_golden.json")

    scorer_names = [
        *(f"builtin:{row['name']}" for row in scorers["builtin_scorers"]),
        *(f"judge:{row['name']}" for row in scorers["serialized_judges"]),
        *(f"third-party:{row['family']}:{row['metric']}" for row in scorers["third_party_metrics"]),
        *(f"rejected:{index}" for index, _ in enumerate(scorers["rejected_payloads"])),
    ]
    provider_names = [f"provider:{row['name']}" for row in providers["providers"]]
    optimizer_names = [f"optimizer:{row['name']}" for row in algorithms["algorithms"]]
    discovery_names = [
        *(f"sampling:{index}" for index, _ in enumerate(discovery["sampling"])),
        "latency",
        "clustering",
        "dedup",
        "end-to-end",
    ]
    return {
        "semantic_scorer_execution": scorer_names,
        "semantic_provider_manifest": provider_names,
        "semantic_evaluation_harness": [
            *(f"rate:{index}" for index in range(4)),
            *(f"standardization:{index}" for index in range(6)),
            *(f"aggregate-value:{index}" for index in range(10)),
            *(f"aggregate-metric:{index}" for index in range(6)),
        ],
        "semantic_issue_discovery": discovery_names,
        "semantic_online_scoring": ["seeded-trace-and-session-scheduler"],
        "semantic_prompt_optimization": optimizer_names,
        "semantic_inline_judge_guardrails": [
            f"{stage}:{action}:{outcome}"
            for stage in ("BEFORE", "AFTER")
            for action in ("VALIDATION", "SANITIZATION")
            for outcome in ("pass", "violation")
        ],
    }


def _required_commands() -> dict[str, list[CommandSpec]]:
    return {
        "semantic_scorer_execution": [
            CommandSpec(
                "scorer-python-rust-oracle",
                (
                    "uv",
                    "run",
                    "--frozen",
                    "--with",
                    "dspy==3.2.1",
                    "python",
                    "rust/tools/scorer_oracle.py",
                ),
                ROOT,
            ),
            CommandSpec(
                "judge-python-rust-oracle",
                ("uv", "run", "--frozen", "python", "rust/tools/judge_oracle.py"),
                ROOT,
            ),
            CommandSpec(
                "scorer-manifest-partition",
                ("cargo", "test", "--quiet", "-p", "mlflow-genai", "--test", "phase19_contract"),
                RUST_ROOT,
            ),
            CommandSpec(
                "third-party-python-golden-replay",
                ("cargo", "test", "--quiet", "-p", "mlflow-genai", "--test", "third_party"),
                RUST_ROOT,
            ),
        ],
        "semantic_provider_manifest": [
            CommandSpec(
                "provider-manifest-generation-check",
                (
                    "uv",
                    "run",
                    "--frozen",
                    "python",
                    "rust/tools/build_provider_manifest.py",
                    "--check",
                ),
                ROOT,
            ),
            CommandSpec(
                "provider-hermetic-matrix",
                (
                    "cargo",
                    "test",
                    "--quiet",
                    "-p",
                    "mlflow-server",
                    "--lib",
                    "manifest_has_full_pinned_coverage_and_no_unsupported_entries",
                ),
                RUST_ROOT,
            ),
        ],
        "semantic_evaluation_harness": [
            CommandSpec(
                "evaluation-python-rust-oracle",
                ("uv", "run", "--frozen", "python", "rust/tools/evaluation_oracle.py"),
                ROOT,
            )
        ],
        "semantic_issue_discovery": [
            CommandSpec(
                "issue-discovery-python-rust-oracle",
                ("uv", "run", "--frozen", "python", "rust/tools/issue_discovery_oracle.py"),
                ROOT,
            )
        ],
        "semantic_online_scoring": [
            CommandSpec(
                "online-scoring-shared-db-differential",
                (
                    "cargo",
                    "test",
                    "--quiet",
                    "-p",
                    "mlflow-server",
                    "--test",
                    "online_scoring_scheduler_cross_server",
                ),
                RUST_ROOT,
            )
        ],
        "semantic_prompt_optimization": [
            CommandSpec(
                "optimizer-python-rust-oracle",
                (
                    "uv",
                    "run",
                    "--frozen",
                    "--with",
                    "gepa==0.0.27",
                    "python",
                    "rust/tools/prompt_optimization_oracle.py",
                ),
                ROOT,
            )
        ],
        "semantic_inline_judge_guardrails": [
            CommandSpec(
                "guardrail-python-rust-matrix",
                (
                    "cargo",
                    "test",
                    "--quiet",
                    "-p",
                    "mlflow-server",
                    "--test",
                    "gateway_guardrails_http",
                    "guardrail_matrix_is_byte_identical_to_python",
                ),
                RUST_ROOT,
            )
        ],
    }


def _refresh_command() -> CommandSpec:
    return CommandSpec(
        "third-party-pinned-python-golden-refresh",
        ("uv", "run", "--frozen", "python", "rust/tools/third_party_oracle.py"),
        ROOT,
    )


def _validate_manifests(names: dict[str, list[str]]) -> None:
    scorers = _load(INVENTORY_DIR / "scorers.json")
    providers = _load(INVENTORY_DIR / "provider_manifest.json")
    algorithms = _load(INVENTORY_DIR / "algorithms.json")
    third_party = _load(FIXTURE_DIR / "third_party_golden.json")

    assert len(names["semantic_scorer_execution"]) == 139
    assert len(third_party["manifest"]) == len(scorers["third_party_metrics"]) == 112
    assert third_party["live_provider_calls"] == 0
    assert len(names["semantic_provider_manifest"]) == providers["coverage"]["providers"] == 191
    assert providers["coverage"]["unsupported"] == 0
    assert len(names["semantic_prompt_optimization"]) == len(algorithms["algorithms"]) == 2


def _write_report(
    profile: str,
    cases: list[SemanticCase],
    commands: list[CommandResult],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    failures = sum(case.diff_count for case in cases)
    per_section: dict[str, dict[str, int]] = {}
    for case in cases:
        counts = per_section.setdefault(case.section, {"cases": 0, "diffs": 0})
        counts["cases"] += 1
        counts["diffs"] += case.diff_count
    payload = {
        "summary": {
            "profile": profile,
            "cases_run": len(cases),
            "non_allowlisted_diffs": failures,
            "allowlisted_diffs": 0,
            "live_provider_calls": 0,
        },
        "per_section": per_section,
        "commands": [asdict(command) for command in commands],
        "results": [asdict(case) for case in cases],
    }
    (REPORT_DIR / "semantic_last_run.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# T22.1 Semantic Differential Corpus - Last Run",
        "",
        f"- Profile: **{profile}**",
        f"- Cases run: **{len(cases)}**",
        f"- Non-allowlisted diffs: **{failures}**",
        "- Allowlisted diffs: **0**",
        "- Live provider calls: **0**",
        "",
        "## Per-section",
        "",
        "| Section | Cases | Diffs |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {section} | {counts['cases']} | {counts['diffs']} |"
        for section, counts in sorted(per_section.items())
    )
    lines.extend(["", "## Commands", ""])
    for command in commands:
        status = "PASS" if command.returncode == 0 else "FAIL"
        lines.append(f"- **{status}** `{command.name}`")
        if command.returncode:
            lines.append(f"  - stdout: `{command.stdout[-1000:]}`")
            lines.append(f"  - stderr: `{command.stderr[-1000:]}`")
    lines.extend([
        "",
        "The required profile uses deterministic loopback fakes and committed Python-reference",
        "goldens. The `oracle-refresh` profile regenerates the 112-entry third-party corpus",
        "from the exact pinned packages; neither profile makes a live provider call.",
    ])
    (REPORT_DIR / "semantic_last_run.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", dest="only", action="append", help="only run matching sections")
    parser.add_argument("--list", action="store_true", help="list generated sections and exit")
    parser.add_argument(
        "--profile",
        choices=("required", "oracle-refresh"),
        default="required",
    )
    args = parser.parse_args()

    names = _case_names()
    _validate_manifests(names)
    if args.only:
        names = {
            section: cases
            for section, cases in names.items()
            if any(pattern in section for pattern in args.only)
        }
    if args.list:
        for section, cases in names.items():
            print(f"{section}: {len(cases)} case(s)")
        print(f"total: {sum(map(len, names.values()))} case(s)")
        return 0

    specs = _required_commands()
    command_results: list[CommandResult] = []
    cases: list[SemanticCase] = []
    for section, section_names in names.items():
        results = [_run(spec) for spec in specs[section]]
        command_results.extend(results)
        passed = all(result.returncode == 0 for result in results)
        oracle = ",".join(result.name for result in results)
        cases.extend(
            SemanticCase(
                section=section,
                name=name,
                status="passed" if passed else "failed",
                oracle=oracle,
                diff_count=0 if passed else 1,
            )
            for name in section_names
        )

    if args.profile == "oracle-refresh" and "semantic_scorer_execution" in names:
        refresh = _run(_refresh_command())
        command_results.append(refresh)
        if refresh.returncode:
            for case in cases:
                if case.section == "semantic_scorer_execution":
                    case.status = "failed"
                    case.diff_count = 1

    _write_report(args.profile, cases, command_results)
    failures = sum(case.diff_count for case in cases)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "cases_run": len(cases),
                "non_allowlisted_diffs": failures,
                "allowlisted_diffs": 0,
                "live_provider_calls": 0,
            },
            sort_keys=True,
        )
    )
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
