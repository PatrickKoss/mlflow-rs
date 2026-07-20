# Rust MLflow Server — Implementation Plan (moved)

This plan was split into multiple files (it grew too large for a single reader)
and now lives in **[`docs/rust-tracking-server-plan/`](docs/rust-tracking-server-plan/)**.

Start at **[`docs/rust-tracking-server-plan/README.md`](docs/rust-tracking-server-plan/README.md)** —
it carries the live status/open block and a map of which file holds which section.

The section (§1–§18), decision (D1–D23), and task (T0.x–T23.x) identifiers are
**unchanged**, so any reference elsewhere in the repo like
`RUST_TRACKING_SERVER_PLAN.md §12.8` or `#T12.4` still resolves — just look up
that section/task in the file the README's map points to:

| You're looking for | File |
|---|---|
| §1–§2 (goal, architecture) | `docs/rust-tracking-server-plan/part1-overview.md` |
| §3–§6 (API surface, wire contract, storage, compliance) | `docs/rust-tracking-server-plan/part1-api-and-contracts.md` |
| §7 + any T0.x–T14.x task | `docs/rust-tracking-server-plan/part1-work-breakdown.md` |
| §8–§10 (verification, decisions D1–D13, sources) | `docs/rust-tracking-server-plan/part1-verification-and-decisions.md` |
| §11–§15 (genai goals, §12 routes, crypto, engines, compliance) | `docs/rust-tracking-server-plan/part2-overview.md` |
| §16 + any T15.x–T23.x task | `docs/rust-tracking-server-plan/part2-work-breakdown.md` |
| §17–§18 (decisions D14–D23, sources) | `docs/rust-tracking-server-plan/part2-decisions-and-appendix.md` |
