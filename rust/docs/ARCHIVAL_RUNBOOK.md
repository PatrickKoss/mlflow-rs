# Trace archival runbook

Server-owned trace archival moves completed trace span payloads out of the SQL
tracking store into a durable artifact repository. Trace metadata remains in
SQL. The trace is marked with `mlflow.trace.spansLocation=ARCHIVE_REPO` and
`mlflow.trace.archiveLocation`; normal trace reads then fetch and decode the
archive's `traces.pb` object.

Use this runbook with [DEPLOYMENT.md](DEPLOYMENT.md). Back up the tracking
database and archive repository as one recovery unit before enabling archival.

## Enable archival

Create a YAML file such as `/etc/mlflow/trace-archival.yaml`:

```yaml
trace_archival:
  enabled: true
  location: file:///mlarchive
  retention: 30d
  long_retention_allowlist:
    - "42"
  interval_seconds: 300
  max_traces_per_pass: 1000
```

`enabled`, `location`, and `retention` are required. Retention accepts a
positive integer followed by `m`, `h`, or `d`. `interval_seconds` defaults to
300 and must be between 1 and 86,400. `long_retention_allowlist` defaults to an
empty list, and an omitted `max_traces_per_pass` is unbounded. Start with an
explicit cap in production.

For the reference compose, add a durable volume and a read-only config mount:

```yaml
services:
  rust:
    environment:
      MLFLOW_TRACE_ARCHIVAL_CONFIG: /etc/mlflow/trace-archival.yaml
    volumes:
      - archive:/mlarchive
      - ./trace-archival.yaml:/etc/mlflow/trace-archival.yaml:ro

volumes:
  archive:
```

The equivalent CLI is
`--trace-archival-config /etc/mlflow/trace-archival.yaml`; the flag overrides
the environment. The file must exist, be readable, and parse successfully
before the listener starts. Archival configuration cannot be combined with
`--artifacts-only`. Keep `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=true`: disabling
job execution also prevents the archival scheduler from starting.

The all-Rust deployment supports local/file and S3-compatible archive
locations. For S3, use the same `AWS_*`, `MLFLOW_S3_ENDPOINT_URL`, TLS, and
addressing-style settings described in [DEPLOYMENT.md](DEPLOYMENT.md), give the
server write/read/delete permission on a dedicated prefix, and test lifecycle
rules against recovery requirements. Each payload is stored at
`<location>/<experiment_id>/traces/<trace_id>/artifacts/traces.pb`.

Deploy privately first and require:

```bash
curl -fsS "$MLFLOW_URL/health"
docker compose -f rust/deploy/docker-compose.yml logs rust \
  | grep 'trace archival scheduler pass completed'
```

An admitted empty pass logs `archived_total=0`, `scope_count`, and
`elapsed_seconds`, proving the enabled scheduler reached the store. Also test a
deliberately invalid file in a disposable container; startup must exit nonzero
with the rejected field. Never perform that negative test against the serving
container.

> **Known signal gap:** the current Rust `/api/3.0/mlflow/server-info` handler
> reports `trace_archival_enabled:false` even when the scheduler is configured.
> Do not use that field as the enablement check; use startup validation, the
> completed-pass log, SQL tags, and archive objects until the handler is fixed.

## Choose retention, interval, and pass size

The scheduler polls immediately at process start and once per minute. The
configured `interval_seconds` admits or skips those polls; setting it below 60
does not create sub-minute polling. Each admitted pass processes archive
uploads sequentially. `max_traces_per_pass` is a successful-archive budget
shared across all workspaces in that pass, and workspace order is shuffled for
fairness.

Start with a retention period comfortably beyond late span writes and incident
investigation needs. Estimate the eligible trace arrival rate, average encoded
span bytes, object-store request latency, and database load. Choose a pass cap
that drains more traces per day than become newly eligible while keeping pass
duration below the configured interval under normal load. Raise the cap or
shorten the interval gradually while watching database, network, and object
store saturation. An omitted cap can turn a first pass over a large existing
database into unbounded work.

All replicas may run the scheduler. The database lock
`trace-archival-scheduler-lock` prevents overlapping passes, so do not add an
external singleton scheduler. Alert if no completed pass appears for more than
two expected intervals, but account for the one-minute poll granularity.

## Monitor progress

Collect these Rust log messages and fields:

- `trace archival scheduler pass completed`: `archived_total`, `scope_count`,
  and `elapsed_seconds`;
- `trace archival scheduler scope failed` and `trace archival scheduler pass
  failed`;
- upload, repository-resolution, finalization, malformed-payload, and archive
  cleanup warnings.

Measure backlog by comparing completed, DB-backed traces older than their
effective retention with archived rows. A basic progress check is:

```sql
SELECT COUNT(*) AS archived_traces
FROM trace_tags
WHERE "key" = 'mlflow.trace.spansLocation'
  AND value = 'ARCHIVE_REPO';

SELECT request_id, value AS archive_location
FROM trace_tags
WHERE "key" = 'mlflow.trace.archiveLocation'
ORDER BY request_id
LIMIT 20;
```

Correlate growth in those rows with object counts and bytes below the archive
prefix. Sample archived traces through both the trace and trace-artifact read
APIs; a database tag alone is not proof that the object remains readable.

The config provider rereads the YAML through a five-second cache. A valid
change is logged when observed. If a later reload is invalid, Rust warns and
continues with the last valid config; fix the file and verify a subsequent pass
instead of assuming the invalid edit disabled archival.

## Force one experiment

Set the `mlflow.trace.archiveNow` experiment tag to a JSON object encoded as the
tag value. `older_than: null` selects all eligible completed traces in the
experiment; a duration limits the request:

```bash
# Archive all completed DB-backed traces for experiment 42 on the next pass.
curl -fsS -H 'Content-Type: application/json' \
  -d '{"experiment_id":"42","key":"mlflow.trace.archiveNow","value":"{\"older_than\":null}"}' \
  "$MLFLOW_URL/api/2.0/mlflow/experiments/set-experiment-tag"

# Or only traces at least 12 hours old.
curl -fsS -H 'Content-Type: application/json' \
  -d '{"experiment_id":"42","key":"mlflow.trace.archiveNow","value":"{\"older_than\":\"12h\"}"}' \
  "$MLFLOW_URL/api/2.0/mlflow/experiments/set-experiment-tag"
```

Archive-now candidates have priority but still consume the shared pass cap.
The scheduler retains the tag while matching traces are in progress or a
retryable failure remains, and clears the exact tag value after processing is
complete. Monitor until the tag clears and verify the expected count and a
sample read. Replacing the tag during a pass is safe: that pass does not delete
the new value.

## Per-experiment retention

Set `mlflow.trace.archivalRetention` to an encoded duration object:

```bash
curl -fsS -H 'Content-Type: application/json' \
  -d '{"experiment_id":"42","key":"mlflow.trace.archivalRetention","value":"{\"type\":\"duration\",\"value\":\"7d\"}"}' \
  "$MLFLOW_URL/api/2.0/mlflow/experiments/set-experiment-tag"
```

An experiment may always shorten the server retention. It may lengthen it only
when its experiment ID appears in `long_retention_allowlist`; otherwise the
server retention wins. Malformed override JSON also falls back to the server
retention. Review the allowlist like a storage-policy exception, and estimate
its database cost before adding an experiment.

## Read and delete semantics

After archival, SQL retains `trace_info`, assessments, and archive-location
tags, but span content and span attributes are removed from SQL. Reads remain
available through the normal APIs and are reconstructed from the archived OTLP
protobuf. This adds the archive repository's latency and availability to trace
reads.

A missing/unreachable archive object currently returns the trace with empty
spans; an empty or malformed object returns an invalid-state corruption error.
Alert on both outcomes. Do not apply an object-store expiration rule that is
shorter than the trace metadata lifecycle. Normal trace deletion deletes the
archive payload before removing the archived trace row; a cleanup failure
leaves the row for a retry rather than silently orphaning metadata.

## Pause, recover, or roll back

To pause new archival, change `enabled` to `false` and wait longer than the
five-second config cache plus the next one-minute poll, or remove the config
setting and restart the server. Confirm that completed-pass logs stop advancing.
Disabling the scheduler does not rehydrate or delete already archived traces;
keep the archive repository mounted and readable.

There is no bulk unarchive/rehydration operation. If archive reads fail:

1. Disable new archival without deleting any objects or SQL tags.
2. Check credentials, DNS, repository permissions, and the exact
   `mlflow.trace.archiveLocation` object first.
3. Restore the object from archive-storage backup when SQL metadata is sound.
4. If consistency cannot be established, stop all writers and restore a
   point-in-time-compatible database **and** archive-repository snapshot into a
   disposable environment before production recovery.

During a Rust-to-Python serving rollback, carry the archival YAML, credentials,
and repository access to the compatible Python service if it will continue
archival. Otherwise disable scheduler ownership before routing changes while
leaving archived reads available. Never let two independently configured
schedulers write different archive roots for the same database.
