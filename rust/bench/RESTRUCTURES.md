# T13.4 deeper-restructure proposals

## Decision context

This document evaluates the four Phase 13.4 candidates against the only measured
T13.3 result set in this repository. That run used SQLite under WSL2, a 2.38 GB
database, 20,000 runs, 8,000,000 metric rows, 50,000 traces, 200,000 spans,
1,000,000 extracted span attributes, 5,000 model versions, and 30 measured
iterations per point scenario. It is not evidence about a 100 GB Postgres database
or concurrent saturation. The full-scale procedure in `rust/bench/README.md` and
`rust/bench/RESULTS.md` must be used before treating any extrapolation as measured.

The relevant T13.3 results are:

| Scenario | Rust p95 | Python p95 | Result relevant here |
|---|---:|---:|---|
| metric-filtered run search | 36.0 ms | 162.3 ms | Rust met the `<500 ms` target |
| deep run-search pagination | 8.5 ms | 36.9 ms | both curves were consistent with O(1) |
| bulk-interval metric history | 3.2 ms | 9.6 ms | 2.99x Rust read-path win |
| span-attribute trace search | 154.6 ms | 265.6 ms | 1.72x, the weakest Rust read-path win |
| OTLP ingest | 2,825.7 spans/s | 675.5 spans/s | 4.18x; the 5x target was missed |
| registered-model prompt exclusion | 86.3 ms | 86.2 ms | p95 tie; the only Rust non-win |

Schema statements below are grounded in
`mlflow/store/tracking/dbmodels/models.py`, the registry and auth DB models, the
T13.1 index revision `a3f8c21d9b47`, and the T13.2 shared `span_attributes`
revision `a8b9c0d1e2f3`. Query and write behavior comes from
`rust/crates/mlflow-store/src/store/`,
`rust/crates/mlflow-registry/src/store/search.rs`, and the Rust auth store and
after-request middleware. DDL is a sketch, not a new migration in this task.

All four proposals have the same non-negotiable compatibility gate. Python and
Rust may share one database during a rollout, both write paths must maintain any
new representation, and responses must remain byte-identical. The T12.4
differential corpus, including its authenticated multi-user cases, must stay green
before, during, and after a read-path switch. A migration that is safe only when
all traffic moves to Rust at once is not acceptable.

The repository contains no measured 100 GB migration throughput. Consequently,
the duration entries below state the amount of work and how to obtain a defensible
ETA; they do not turn the 2.38 GB SQLite seed time into an invented Postgres
estimate. Every backfill must report rows and bytes per second after each batch,
persist a high-water mark, and compute `remaining rows / observed rate` on the
target system. The Alembic transaction should contain only short catalog changes;
the resumable data copy belongs in an operator-run backfill command.

## 1. Hash-partition `metrics` by run

### Problem and evidence

`metrics` has a six-column primary key over `(key, timestamp, step, run_uuid,
value, is_nan)`, an index on `run_uuid`, and an index on `(run_uuid, key, step)`.
The Rust history paths in `store/metrics.rs` and `store/metrics_bulk.rs` constrain
by one or at most 100 run IDs and a metric key; interval history also obtains
distinct steps, per-run min/max steps, and rows for selected steps. Hash
partitioning on `run_uuid` matches those access patterns. It does not directly
improve metric-filtered run search, which reads the already compact
`latest_metrics` table rather than full history.

There is no measured need at the tested scale. Eight million history rows produced
a 3.2 ms Rust p95 for bulk-interval history, and metric-filtered run search was
36.0 ms p95 against a 500 ms target. Deep pagination also stayed flat. These
numbers argue against paying partition-management and write-routing costs now.
They say nothing about hundreds of millions or billions of metric rows, partition
pruning on Postgres, or index working-set pressure under concurrent training.

### Design

If full-scale Postgres proves that the unpartitioned history indexes are the
bottleneck, use fixed-count hash partitioning by `run_uuid`. Time partitioning is
a poor fit: every normal history query starts with a run, retention is not part of
the current wire contract, and one run can span arbitrary timestamps.

Illustrative Postgres DDL (names and partition count require a measured design
review):

```sql
CREATE TABLE metrics_partitioned (
  key VARCHAR(250) NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  timestamp BIGINT NOT NULL,
  step BIGINT NOT NULL,
  is_nan BOOLEAN NOT NULL,
  run_uuid VARCHAR(32) NOT NULL REFERENCES runs(run_uuid),
  CONSTRAINT metric_partitioned_pk
    PRIMARY KEY (key, timestamp, step, run_uuid, value, is_nan)
) PARTITION BY HASH (run_uuid);

CREATE TABLE metrics_partitioned_p00 PARTITION OF metrics_partitioned
  FOR VALUES WITH (MODULUS 32, REMAINDER 0);
-- Repeat for remainders 1..31.

CREATE INDEX index_metrics_partitioned_run_uuid
  ON metrics_partitioned (run_uuid);
CREATE INDEX index_metrics_partitioned_run_uuid_key_step
  ON metrics_partitioned (run_uuid, key, step);
```

The partition count is deliberately not a recommendation. Select it from the
100 GB Postgres plans and concurrent soak; too many partitions make the bulk API,
which accepts up to 100 run IDs, pay planning and fan-out costs.

The SQL predicates and response ordering do not change. `get_metric_history`,
bulk history, distinct-step, min/max-step, selected-step, and run-deletion queries
point at the new logical table. Both servers continue sanitizing NaN/Inf and using
the existing six-column key for deduplication. `latest_metrics` is not partitioned.

This should be Postgres-only unless another supported dialect demonstrates a need
and a safe implementation. SQLite has no native partitioning. MySQL partitioning
has materially different key and foreign-key constraints. Building a portable
UNION view over hand-sharded tables would force every Python and Rust write and
delete through custom routing and is not justified by T13.3.

Write-path implication: partition routing is database-side, so steady-state
Python and Rust inserts retain their current SQL. During migration, however, both
must write the old table and let a database trigger mirror inserts/deletes to the
shadow partitioned table, or both codebases must dual-write. The trigger is safer
for genuinely mixed versions because an older server cannot know about the shadow
table. Its write amplification must be measured in the T14.2 training workload.

### Migration

Use two phases rather than replacing `metrics` in one Alembic transaction.

1. A shared tracking-store revision creates `metrics_partitioned`, its children,
   matching indexes and foreign key, plus idempotent mirror triggers from
   `metrics`. The existing table remains authoritative. Upgrade is additive and
   current Python and Rust binaries keep working.
2. A separate resumable backfill keyset-walks the current primary key in bounded
   batches and inserts with conflict-ignore. Store the last copied key and batch
   checksum in a migration-progress table. Concurrent old-table writes are
   captured by the mirror trigger; a final anti-join/count reconciliation proves
   that both representations contain the same keys.
3. A capability flag allows new Python and Rust binaries to read the partitioned
   table while writes still enter through `metrics`. Run T12.4 against both read
   choices and run the full T13.3 Postgres suite.
4. Only after every process understands the new table may a later revision make
   it authoritative and retire the mirror. Keep the old table for a rollback
   window. A physical rename is optional; explicit table selection is clearer
   during the compatibility period.

At 100 GB this is at least one full read of `metrics`, one full write of the table
and all indexes, plus temporary double storage and ongoing mirrored writes. No
repository measurement supports an hour estimate. Run representative batches on
the restored 100 GB seed, record bytes/second and WAL growth, and use the measured
rate for the maintenance ETA. Abort before enabling the mirror if free space
cannot hold both copies, their indexes, and WAL/replication lag.

### Rollback

Before the read switch, disable and drop the mirror trigger and shadow table; the
old table never stopped being authoritative. After the read switch, flip both
servers back to `metrics`, reconcile any rows that could have entered only the
shadow representation, then disable the trigger. If a final cutover has already
made the partitioned table authoritative, reverse-mirror into the retained old
table and verify key equality before switching reads/writes back. Never drop the
old table in the same release that changes reads.

### Verdict: DEFER

Revisit only when the 100 GB Postgres run (tuned to approximately the README's
750,000 runs x 5 keys x 100 points) or the T14.2 one-hour Postgres soak shows a
material post-training history latency regression under load, or sustained I/O /
cache pressure attributable by `EXPLAIN (ANALYZE, BUFFERS)` and Postgres statistics
to the history indexes. The 500 ms metric-filtered run-search SLO is not a trigger
for this change because that query uses `latest_metrics`, not the partition
candidate. Partitioning must beat the unpartitioned schema on both history
p95/p99 and training ingest after including mirror/partition overhead.

## 2. Replace the wide metrics primary key with a dedup hash

### Problem and evidence

The six-column `metric_pk` is both an identity and a dedup mechanism. It includes
two strings and a floating-point value, so every primary-key entry is wider than
a fixed digest. Rust relies on it directly: `metrics_insert_sql` performs
conflict-ignore on all six columns, matching Python's observable behavior. No
table references a metric history row by this primary key, so a narrower physical
identity is possible without changing API entities.

T13.3 did not measure primary-key bytes, WAL, page splits, insert CPU, or cache-hit
rate. Its only directly relevant read was already 3.2 ms p95 over 8 million rows.
The run-search result mostly exercises `latest_metrics`, whose two-column primary
key is not part of this proposal. There is therefore no measured performance case
for changing the dedup identity now.

### Design

Use a 32-byte SHA-256 digest of the canonical stored tuple as the primary key,
while retaining every existing column and the two history indexes:

```sql
CREATE TABLE metrics_hashed (
  dedup_hash BINARY(32) NOT NULL,
  key VARCHAR(250) NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  timestamp BIGINT NOT NULL,
  step BIGINT NOT NULL,
  is_nan BOOLEAN NOT NULL,
  run_uuid VARCHAR(32) NOT NULL REFERENCES runs(run_uuid),
  CONSTRAINT metric_hashed_pk PRIMARY KEY (dedup_hash)
);

CREATE INDEX index_metrics_hashed_run_uuid
  ON metrics_hashed (run_uuid);
CREATE INDEX index_metrics_hashed_run_uuid_key_step
  ON metrics_hashed (run_uuid, key, step);
```

`BINARY(32)` is illustrative; Postgres would use `BYTEA` and SQLite `BLOB`.
Digest input must be specified byte-for-byte, not as database string formatting:
version byte; length-prefixed UTF-8 `key`; signed big-endian 64-bit timestamp and
step; length-prefixed ASCII `run_uuid`; canonical IEEE-754 value bits; and one
boolean byte. Apply the existing storage normalization first (NaN becomes
`value=0.0,is_nan=true`, infinities clamp), and canonicalize both signed zeros to
positive zero if differential tests confirm the current SQL unique semantics
treat them as equal. Python and Rust need shared golden vectors for ordinary,
Unicode, NaN, infinity, and signed-zero cases.

All history SELECTs keep their current projections, filters, and ordering; the
digest never appears on the wire. Both writers compute the same digest and use
conflict-ignore on it. A full SHA-256 collision would turn two distinct metric
rows into one, unlike the current key. The risk is remote but the semantic change
is real. An implementation must detect a conflict whose stored six columns differ
and fail closed with an internal consistency error rather than silently claim it
was a duplicate. That preserves observability of corruption, though it cannot
make a digest-only unique key mathematically identical to tuple uniqueness.

### Migration

Use an additive compatibility revision first:

1. Add a nullable `dedup_hash` to the current table and create a non-unique index,
   or create the shadow `metrics_hashed` table shown above. A shadow table is more
   portable because replacing primary keys commonly rebuilds the whole table.
2. Release Python and Rust writers that compute the same hash. During a mixed
   deployment, retain the old six-column primary key as the authority; new writers
   populate the digest, while old writers may leave it null. A database mirror
   trigger or a catch-up worker computes missing hashes so old binaries remain
   compatible. Do not make the column non-null yet.
3. Backfill in primary-key order in resumable batches. For each batch, compare row
   count, all-column checksum, and duplicate/collision classification. Resume from
   the persisted six-column high-water key.
4. After no old writer remains and all rows have a verified digest, T12.4 passes
   against the hashed read/write path, and a shadow benchmark shows a material
   benefit, a second revision builds the digest primary key and removes the old
   one. This is the point that requires a table rebuild on several dialects.

At 100 GB the shadow-table form reads every metric, hashes every row, writes a
second copy, and builds three indexes; the in-place form still scans the table and
rebuilds the primary index. Duration is unknown from repository data. Measure
hash/backfill rows per second and index-build/WAL time on a clone of the full-scale
Postgres seed, then calculate the ETA from those observations. Plan for double
storage in the shadow form and for a long-running table rewrite in dialects that
cannot replace the primary key online.

### Rollback

Before the primary-key switch, stop digest reads, leave the old key in force, and
drop the shadow table or nullable column after the rollback window. Partially
backfilled rows are harmless. After the switch, keep the old tuple columns and a
verified old-format shadow table. Dual-write/reverse-backfill rows created since
cutover, compare the full six-column set, switch reads and conflict handling back,
and only then remove the digest key. A digest collision discovered at any stage is
an automatic rollback trigger, not a row to discard.

### Verdict: DEFER

Revisit when the 100 GB Postgres run records `pg_relation_size` /
`pg_indexes_size`, WAL bytes, buffer hit rate, and metric insert throughput and
shows the wide primary index is a leading storage or write-amplification cost.
The decision requires an A/B shadow schema proving that the fixed hash materially
reduces total table-plus-index bytes or improves saturated metric ingest without
regressing the 3.2 ms-class history path or changing dedup behavior. T13.3 contains
none of those measurements.

## 3. Split spans into hot metadata and cold payload tables

### Problem and evidence

`spans` stores searchable scalar columns, the stored generated `duration_ns`,
`dimension_attributes`, and the full JSON `content` text in one row. T13.2 added
`span_attributes(trace_id, span_id, key, value, value_truncated)` with a
`(key,value)` index. Rust's common case-sensitive attribute LIKE path now uses that
table as a prefilter but intentionally retains `s.content LIKE ...` as a residual
predicate to preserve Python's JSON substring quirks. Trace response hydration
also selects full content, and archival semantics use `content = ''` to mean the
payload has been cleared.

Span-attribute search was the weakest Rust read win, 154.6 ms p95 and only 1.72x
Python, at 200,000 spans and 1,000,000 extracted attributes. That makes this the
most plausible of the four physical restructures to study. It still does not prove
a split is needed: the measured p95 is below T13.2's `<1 s` full-scale target, but
that target is for 50 million spans on Postgres and has not been run. The same
T13.3 run also missed the OTLP 5x target, yet splitting one insert into hot and cold
rows adds write work and could make ingest worse.

### Design

Keep identity and searchable columns in `spans`; move only payload text into a
one-to-one table. Keep `dimension_attributes` hot initially because it participates
in analytics and is much smaller than arbitrary span content.

```sql
CREATE TABLE span_payloads (
  trace_id VARCHAR(50) NOT NULL,
  span_id VARCHAR(50) NOT NULL,
  content TEXT NOT NULL,
  PRIMARY KEY (trace_id, span_id),
  CONSTRAINT fk_span_payloads_span
    FOREIGN KEY (trace_id, span_id)
    REFERENCES spans(trace_id, span_id) ON DELETE CASCADE
);
```

Search predicates on name/type/status/duration and the `span_attributes`
prefilter stay on the hot tables. For the compatibility-sensitive attribute LIKE
case, join `span_payloads` only for candidate rows and apply the same residual LIKE
to `span_payloads.content`. Trace selection should find and page trace IDs before
loading payloads, then hydrate the selected traces in a batched join. Content/text,
ILIKE, RLIKE, wildcard/escaped attribute keys, and long-key fallbacks still require
payload access and must retain their exact dialect semantics.

Both Python `log_spans` and Rust's shared `log_spans`/OTLP path must upsert the hot
row, cold row, extracted attributes, span metrics, and trace aggregates in the same
transaction. Relogging replaces the cold content and extracted attributes together.
Trace/workspace deletion relies on cascade. Archival must set the cold payload to
`''` (or delete it only if reads preserve the current cleared-payload behavior) and
must not leave the old hot copy readable. The serialized response is assembled from
the same content bytes, so no JSON re-encoding is allowed. T12.4 needs adversarial
content, archival, relog, concurrent start/log, and all fallback comparators.

### Migration

Use a two-revision expansion/contraction sequence like T13.2's shared D7
resolution, but retain the old column much longer:

1. Expansion revision creates `span_payloads` and its cascade FK. Current
   `spans.content` remains non-null and authoritative. Release Python and Rust
   dual-write support. For mixed deployments that can include an older binary,
   an idempotent trigger mirrors every insert/update of `spans.content`; otherwise
   a stale payload row could make new readers return old bytes.
2. A resumable keyset backfill walks `(trace_id,span_id)` in bounded batches and
   upserts the content verbatim. Persist the high-water key, reconcile missing /
   unequal payloads, and let the trigger cover concurrent writes. The backfill
   must preserve `''` exactly.
3. Enable cold-table reads behind a capability flag with fallback to
   `spans.content` only for a demonstrably missing shadow row. Run the T12.4 corpus
   and full-scale Postgres trace scenarios against both choices. Do not remove the
   residual predicate.
4. Only after all Python and Rust processes dual-write and the rollback window has
   passed may a contraction revision rebuild `spans` without `content`. That final
   contraction creates the actual cache/storage benefit and is incompatible with
   old binaries, so it needs a separately announced minimum server version.

At 100 GB the expansion scans every span row and writes the payload bytes again,
temporarily duplicating the largest column and generating corresponding WAL. The
final contraction rewrites the hot table. There is no measured repository rate
from which to claim a duration. Measure copied payload bytes/second—not just
rows/second—on the full-scale Postgres clone and forecast from remaining payload
bytes. Throttle batches when replica lag or T14.2 ingest p95 rises.

### Rollback

Before contraction, flip reads back to `spans.content`; it remained authoritative.
Disable the mirror only after all dual-writing binaries are rolled back, then drop
`span_payloads` if desired. Partial backfill is safe because no old reads changed.
After contraction, reverse the expansion: add nullable `spans.content`, backfill
verbatim from `span_payloads`, make it non-null after equality checks, deploy
readers that use it, and only then drop the cold table. Keep archival markers and
delete cascades active throughout.

### Verdict: DEFER

First run the documented full-scale Postgres procedure, increasing `--traces` /
`--spans-per-trace` until it contains the T13.2 target of 50 million spans (the
README's initial 2,000,000 x 5 candidate produces 10 million), and run the existing
span-filter scenario for 200 iterations after `ANALYZE`. Add that same filter to
the concurrent T14.2 trace-ingest workload. Revisit if the current schema misses
the `<1 s` p95 target and `EXPLAIN (ANALYZE, BUFFERS)` attributes the time to
fetching wide `spans` heap pages/content after the `span_attributes` prefilter.
Also require a cold-split A/B to show that any read gain survives the extra OTLP
writes; the current 4.18x ingest result gives no budget for an unmeasured write
regression.

## 4. Integrate auth grants as a search semi-join

### Problem and evidence

Auth grants live in a separate auth database by default. The live schema is
`roles`, `role_permissions`, and `user_role_assignments`; roles are scoped by
workspace, and `role_permissions` is unique on `(role_id, resource_type,
resource_pattern)`. Per-user grants are represented by synthetic roles. Rust's
`readable_set` first joins those auth tables through
`list_role_grants_for_user_in_workspace`, then filters search responses in memory.
For experiments, registered models, and logged models it may repeatedly fetch
later store pages to refill `max_results`. Model-version filtering is drop-only,
matching Python.

T13.3 did not enable or measure authenticated sparse-grant searches. The
registered-model prompt result is not auth evidence: it measures prompt exclusion
from registry tags. There is no measured refetch count, auth-query time, or
authorized-search p95 to motivate grant materialization. Deep pagination being
O(1) without auth does not settle the page-refill behavior.

### Design

Do not persist effective grants in the tracking database. The auth database may be
separate and may have its own read replica; duplicating security state into every
backend store creates a cross-database consistency window in which revoked access
could remain readable. Instead, materialize the already resolved readable resource
patterns as a request-local derived table and semi-join it into the existing search
query:

```sql
-- Illustrative Postgres/SQLite shape; render VALUES/casts per dialect.
WITH readable(resource_id) AS (
  VALUES (?), (?), (?)
)
SELECT e.*
FROM experiments e
WHERE e.workspace = ?
  AND EXISTS (
    SELECT 1 FROM readable g WHERE g.resource_id = CAST(e.experiment_id AS VARCHAR)
  )
-- existing filter and ordering
LIMIT ?;
```

The same shape applies to `registered_models.name` and
`logged_models.experiment_id`. A readable `*` grant or the single-tenant default
READ fallback bypasses the semi-join. A user with no readable grant gets a false
predicate. For a large grant set, use a connection-local temporary table within
the search transaction only after measuring parameter/planner limits; never use a
persistent cache as the first implementation.

The auth resolution itself remains authoritative and unchanged: max-merge matching
role permissions, workspace `MANAGE` folding, `NO_PERMISSIONS` boundary, prompt vs
registered-model namespace, admin bypass, and default-permission fallback all run
before constructing the derived table. The store applies permission filtering
before `LIMIT`, so one query naturally fills a page. Pagination tokens must still
describe the filtered stream exactly. Model-version search must preserve Python's
drop-only/token behavior unless the differential corpus explicitly proves that an
integrated form is byte-identical.

There is no write-path change for either server: role, assignment, grant, revoke,
rename, delete, and creator-grant operations continue to write only the auth DB.
Each request observes the auth store/read-replica state it already would have used.
This avoids the unsafe two-database dual-write problem.

### Migration

No Alembic revision is needed for the preferred request-local design. It is a
query-only expansion behind the existing
`MLFLOW_RUST_AUTH_QUERY_INTEGRATED_FILTERING` seam documented in
`auth_middleware/after_request.rs`; the Python-identical refetch path stays the
default and fallback. Implement and test each resource type separately, compare
ordered entities and tokens against the refetch form, then run the authenticated
T12.4 corpus on SQLite and Postgres.

If measurements eventually require a persistent effective-grant table, that is a
different proposal: it would need an auth Alembic revision, Python and Rust writers
for every role/grant/assignment mutation, generation numbers, revocation-before-
read guarantees, and a distribution mechanism into potentially separate tracking,
registry, and workspace databases. Without a transactional cross-database protocol,
that persistent variant should be rejected on security grounds.

At 100 GB there is no data backfill for the request-local form; rollout duration is
application deployment plus corpus validation. Query cost depends on grants per
user, not backend database size. Benchmark users with zero, sparse, wildcard, and
large role-derived grant sets and report auth DB time, bound-ID count, refill query
count, and end-to-end p95/p99.

### Rollback

Disable the capability flag and return immediately to the existing after-request
filter/refetch implementation. No database state needs repair. Keep the fallback
for at least one release after enabling the semi-join. Any mismatch in entity order,
page contents, next-page token, workspace boundary, or prompt classification is an
automatic flag rollback.

### Verdict: DEFER

Add an authenticated benchmark before implementation. Revisit when a sparse-grant
version of the T13.3 searches or the T14.2 soak shows repeated page-refill queries
and materially worse authorized p95/p99 than the same underlying search for an
admin, with query counts identifying the refetch loop as the cause. The persistent
cross-database materialization variant is rejected; only the request-local
semi-join is a viable deferred optimization.

## 5. Findings outside the four candidates

### Prompt anti-join tie: isolate query shape from hydration

The prompt-exclusion scenario requests 100 registered models ordered by name and
checks that prompt-prefixed rows are absent. The schema has no dedicated prompt
flag: `registered_model_tags` has primary key `(workspace,key,name)`, and both
servers preserve Python's LEFT JOIN against a grouped prompt-tag subquery followed
by `prompts.name IS NULL`. Rust p95 was 86.3 ms versus Python 86.2 ms, the only
non-win. This is evidence to investigate, not evidence that the LEFT JOIN is the
bottleneck.

The measured request includes entity hydration. Rust's
`search_registered_models` loops over the selected names and calls
`get_registered_model` once per model to load the full entity. Python eager-loads
relationships and batch-fetches latest versions for all selected model names.
It is therefore plausible that Rust's per-model queries, not the anti-join, account
for much of the flat Rust latency; this is a code-grounded hypothesis, not a
benchmark conclusion. The Python distribution also had a 24.4 ms p50 but 86.2 ms
p95, while Rust was roughly flat at 84.2/86.3 ms, so p95 alone hides different
cost shapes.

Before a schema change, capture query counts and separately time: (1) IDs from the
LEFT JOIN form, (2) an equivalent `NOT EXISTS`, and (3) full hydration. On the
100 GB Postgres clone, compare plans and buffers for the current tag primary key,
a Postgres candidate `(workspace,key,value,name)` index, and `NOT EXISTS`. The
5,000-character tag value makes that full portable index unsuitable for some
dialects, so no cross-dialect migration should be inferred from the experiment.
Preserve default prompt exclusion and explicit prompt-filter semantics exactly. If
hydration is the cost, batch tags, aliases, and latest versions in Rust; that is a
query-path change, not one of the four deeper restructures.

### OTLP 4.18x result: profile batching before changing schema

The OTLP scenario posts sequential batches of 100 new spans. Rust achieved
2,825.7 spans/s versus Python's 675.5 spans/s, but missed the 5x target. The result
does not distinguish protobuf translation, SQL statement overhead, lock time,
WAL/fsync, or storage layout.

Rust already uses one transaction for the request, so per-span fsync is not shown
by the code. Within that transaction, however, `log_spans` loops over spans; each
span performs a span upsert, deletes its old extracted attributes, then upserts
each extracted string attribute separately. It then updates trace aggregates and
location tags per trace. The benchmark payload contains new traces and two string
attributes per span. It is plausible that statement count and database round trips
limit the result. That remains speculation until tracing separates translation,
pool acquisition, statement execution, commit/fsync, and attribute extraction.

The next experiment should compare prepared multi-row span/attribute upserts and
set-based trace updates using the same wire payload and one transaction, first on
the local corpus and then under concurrent T14.2 Postgres load. Record commit time,
statements/request, WAL bytes, spans/s, and response parity. A hot/cold split should
not be used to explain or fix the 4.18x gap unless profiling specifically identifies
wide-row writes as the limiting cost.
