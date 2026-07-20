#!/usr/bin/env python3
"""Classify first-parent upstream drift from the recorded Rust sync anchor."""

import argparse
import fnmatch
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BUCKET_PRIORITY = ("server-api", "ui", "client-sdk", "infra")

# Keep path policy declarative and near the top of the file. A path may match more
# than one bucket; the commit receives the highest-priority bucket and reports all
# matches as ``mixed``. Add exact paths before broad prefixes where practical.
CLASSIFICATION = {
    "server-api": {
        "exact": (
            "mlflow/server/handlers.py",
            "mlflow/tracing/archival.py",
            "mlflow/tracing/trace_archival_config.py",
            "mlflow/tracing/trace_archival_service.py",
            "mlflow/genai/scheduled_scorers.py",
        ),
        "prefixes": (
            "mlflow/server/auth/",
            "mlflow/server/jobs/",
            "mlflow/server/assistant/",
            "mlflow/assistant/",
            "mlflow/store/",
            "mlflow/protos/",
            "mlflow/entities/",
            "mlflow/webhooks/",
            "mlflow/gateway/",
            "mlflow/deployments/server/",
            "mlflow/tracing/otel/otel_archival.py",
            "mlflow/genai/scorers/",
            "mlflow/genai/judges/",
            "mlflow/genai/evaluation/",
            "mlflow/genai/optimize/",
            "mlflow/genai/discovery/",
            "mlflow/genai/prompts/",
            "mlflow/genai/gateway/",
            "mlflow/genai/review_queues/",
            "mlflow/genai/label_schemas/",
            "mlflow/utils/model_catalog/",
        ),
        "globs": (
            # These UI files define server-persisted/ajax wire formats, not only rendering.
            "mlflow/server/js/**/savedViewEnvelope.*",
        ),
    },
    "ui": {
        "prefixes": ("mlflow/server/js/",),
    },
    "client-sdk": {
        "exact": (
            "mlflow/sklearn.py",
            "mlflow/xgboost.py",
            "mlflow/lightgbm.py",
            "mlflow/catboost.py",
            "mlflow/h2o.py",
            "mlflow/fastai.py",
            "mlflow/gluon.py",
            "mlflow/statsmodels.py",
            "mlflow/spacy.py",
            "mlflow/prophet.py",
            "mlflow/pmdarima.py",
            "mlflow/ml-package-versions.yml",
            "mlflow/ml_package_versions.py",
            "mlflow/utils/uri.py",
        ),
        "prefixes": (
            "mlflow/tracking/",
            "mlflow/pyfunc/",
            "mlflow/models/",
            "mlflow/pytorch/",
            "mlflow/keras/",
            "mlflow/tensorflow/",
            "mlflow/transformers/",
            "mlflow/sentence_transformers/",
            "mlflow/onnx/",
            "mlflow/mxnet/",
            "mlflow/spark/",
            "mlflow/anthropic/",
            "mlflow/bedrock/",
            "mlflow/dspy/",
            "mlflow/gemini/",
            "mlflow/groq/",
            "mlflow/litellm/",
            "mlflow/mistral/",
            "mlflow/autogen/",
            "mlflow/crewai/",
            "mlflow/openai/",
            "mlflow/langchain/",
            "mlflow/llama_index/",
            "mlflow/pydantic_ai/",
            "mlflow/pyspark/",
            "mlflow/shap/",
            "mlflow/utils/autologging/",
            # Tracing export/instrumentation is client-side unless a server rule above
            # also matches it.
            "mlflow/tracing/",
        ),
        "globs": (
            "mlflow/**/autolog.py",
            "mlflow/**/autologging.py",
        ),
    },
    "infra": {
        "exact": ("uv.lock",),
        "prefixes": (".github/", "docs/", "tests/", "dev/", "libs/"),
    },
}

# Server modules outside the named subtrees are server-facing by definition. Keep
# this separate so ``mlflow/server/js`` remains a UI rule except for contract files.
SERVER_MODULE_PREFIX = "mlflow/server/"
SERVER_MODULE_EXCLUDE = "mlflow/server/js/"


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str
    files: tuple[str, ...]
    bucket: str
    mixed: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "sha": self.sha,
            "short_sha": self.sha[:10],
            "subject": self.subject,
            "files_touched": len(self.files),
            "key_paths": key_paths(self),
            "bucket": self.bucket,
            "mixed": list(self.mixed),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--state", type=Path, default=Path("rust/sync/state.json"))
    parser.add_argument("--upstream-ref", default="upstream/master")
    parser.add_argument(
        "--fetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fetch upstream master before analysis (default: fetch)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write drift-report.md in this directory (default: stdout only)",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--fail-on-relevant", action="store_true")
    return parser.parse_args()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def resolve_from_repo(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def matches_rule(path: str, rule: dict[str, tuple[str, ...]]) -> bool:
    return (
        path in rule.get("exact", ())
        or any(path.startswith(prefix) for prefix in rule.get("prefixes", ()))
        or any(fnmatch.fnmatchcase(path, pattern) for pattern in rule.get("globs", ()))
    )


def buckets_for_path(path: str) -> set[str]:
    buckets = {bucket for bucket, rule in CLASSIFICATION.items() if matches_rule(path, rule)}
    if path.startswith(SERVER_MODULE_PREFIX) and not path.startswith(SERVER_MODULE_EXCLUDE):
        buckets.add("server-api")
    return buckets or {"infra"}


def classify_commit(sha: str, subject: str, files: tuple[str, ...]) -> Commit:
    buckets = set().union(*(buckets_for_path(path) for path in files)) if files else {"infra"}
    ordered = tuple(bucket for bucket in BUCKET_PRIORITY if bucket in buckets)
    return Commit(
        sha=sha,
        subject=subject,
        files=files,
        bucket=ordered[0],
        mixed=ordered if len(ordered) > 1 else (),
    )


def read_commits(repo: Path, anchor: str, upstream_ref: str) -> list[Commit]:
    log = git(
        repo,
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H%x09%s",
        f"{anchor}..{upstream_ref}",
    )
    commits = []
    for line in log.splitlines():
        sha, subject = line.split("\t", 1)
        names = git(
            repo,
            "show",
            "--format=",
            "--name-only",
            "--first-parent",
            "--diff-merges=first-parent",
            "--diff-filter=ACDMRT",
            sha,
        )
        files = tuple(dict.fromkeys(path for path in names.splitlines() if path))
        commits.append(classify_commit(sha, subject, files))
    return commits


def key_paths(commit: Commit, limit: int = 4) -> list[str]:
    primary = [path for path in commit.files if commit.bucket in buckets_for_path(path)]
    remaining = [path for path in commit.files if path not in primary]
    return (primary + remaining)[:limit]


def markdown_report(
    *, anchor: str, upstream_ref: str, upstream_head: str, commits: list[Commit]
) -> str:
    counts = Counter(commit.bucket for commit in commits)
    lines = [
        "# Upstream drift report",
        "",
        f"- Sync anchor: `{anchor}`",
        f"- Upstream ref: `{upstream_ref}` at `{upstream_head}`",
        f"- First-parent commits since anchor: **{len(commits)}**",
        f"- Rust-relevant server API commits: **{counts['server-api']}**",
        "",
        "## Bucket counts",
        "",
        "| Bucket | Commits |",
        "|---|---:|",
    ]
    lines.extend(f"| `{bucket}` | {counts[bucket]} |" for bucket in BUCKET_PRIORITY)

    for bucket in BUCKET_PRIORITY:
        lines.extend([
            "",
            f"## {bucket}",
            "",
            "| Commit | Subject | Files | Key paths | Mixed |",
            "|---|---|---:|---|---|",
        ])
        bucket_commits = [commit for commit in commits if commit.bucket == bucket]
        if not bucket_commits:
            lines.append("| — | No commits | 0 | — | — |")
            continue
        for commit in bucket_commits:
            paths = "<br>".join(f"`{path}`" for path in key_paths(commit)) or "—"
            mixed = ", ".join(f"`{item}`" for item in commit.mixed) or "—"
            subject = commit.subject.replace("|", "\\|")
            lines.append(
                f"| `{commit.sha[:10]}` | {subject} | {len(commit.files)} | {paths} | {mixed} |"
            )
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    repo = args.repo.resolve()
    if args.fetch:
        git(repo, "fetch", "upstream", "master")

    state_path = resolve_from_repo(repo, args.state)
    state = json.loads(state_path.read_text())
    anchor = state["last_synced_upstream_commit"]
    upstream_head = git(repo, "rev-parse", args.upstream_ref)
    commits = read_commits(repo, anchor, args.upstream_ref)
    counts = Counter(commit.bucket for commit in commits)
    report = markdown_report(
        anchor=anchor,
        upstream_ref=args.upstream_ref,
        upstream_head=upstream_head,
        commits=commits,
    )
    payload = {
        "anchor": anchor,
        "upstream_ref": args.upstream_ref,
        "upstream_head": upstream_head,
        "commits_total": len(commits),
        "commits_relevant": counts["server-api"],
        "counts": {bucket: counts[bucket] for bucket in BUCKET_PRIORITY},
        "commits": [commit.as_json() for commit in commits],
        "markdown": report,
    }
    return payload, report


def main() -> int:
    args = parse_args()
    try:
        payload, report = analyze(args)
        if args.output_dir:
            output_dir = resolve_from_repo(args.repo.resolve(), args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "drift-report.md").write_text(report)
        sys.stdout.write(json.dumps(payload, indent=2) if args.json_output else report)
        return int(args.fail_on_relevant and payload["commits_relevant"] > 0)
    except (OSError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        sys.stderr.write(f"upstream drift analysis failed: {error}\n")
        # Drift is informational by default. CI that wants a status signal uses
        # --fail-on-relevant; diagnostics still go to stderr for operational errors.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
