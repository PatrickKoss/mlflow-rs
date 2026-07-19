"""Run Python's archive→read→delete path and emit a normalized parity record."""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine, text

from mlflow.entities import trace_location
from mlflow.entities.span import Span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.store.tracking.dbmodels.models import SqlSpan, SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import TraceTagKey


def _normalize_live_default_experiment(backend_uri: str) -> None:
    if backend_uri.startswith("sqlite"):
        return
    engine = create_engine(backend_uri)
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE experiments SET experiment_id = 0 "
                "WHERE workspace = 'default' AND name = 'Default' "
                "AND NOT EXISTS (SELECT 1 FROM experiments WHERE experiment_id = 0)"
            )
        )
    engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", required=True)
    args = parser.parse_args()

    trace_id = "tr-00112233445566778899aabbccddeeff"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive_root = root / "archive"
        archive_root.mkdir()
        backend_uri = os.environ.get(
            "MLFLOW_TRACE_ARCHIVAL_PYTHON_BACKEND_URI",
            f"sqlite:///{root / 'tracking.db'}",
        )
        # TempDb recreates the shared live fixture through Rust's normal store
        # API, which gives Default a generated ID. Python requires its ID to be
        # exactly zero during SqlAlchemyStore initialization.
        _normalize_live_default_experiment(backend_uri)
        store = SqlAlchemyStore(backend_uri, (root / "mlruns").as_uri())
        # The live dialect matrix shares one sequential test database. Remove
        # only this differential's fixed golden trace ID if a prior case left it
        # behind; every other row remains owned by the surrounding test fixture.
        with store.ManagedSessionMaker(read_only=False) as session:
            session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).delete()
        experiment_id = store.create_experiment(f"archive-differential-{uuid.uuid4().hex}")
        store.start_trace(
            TraceInfo(
                trace_id=trace_id,
                trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
                request_time=0,
                execution_duration=1,
                state=TraceState.OK,
                tags={},
                trace_metadata={},
            )
        )
        store.log_spans(experiment_id, [Span.from_dict(json.loads(args.content))])

        with mock.patch.object(store, "_get_archive_traces_now_millis", return_value=60_000):
            archived = store.archive_traces(
                resolved_trace_archival_location=archive_root.as_uri(),
                broader_retention="1m",
                max_traces_per_pass=10,
            )

        info = store.get_trace_info(trace_id)
        archive_uri = info.tags[TraceTagKey.ARCHIVE_LOCATION]
        payload_path = Path(archive_uri.removeprefix("file://")) / "traces.pb"
        payload = payload_path.read_bytes()
        read_json = store.get_trace(trace_id, allow_partial=True).data.to_dict()
        with store.ManagedSessionMaker() as session:
            stored_content = (
                session.query(SqlSpan.content).filter(SqlSpan.trace_id == trace_id).one()[0]
            )
        deleted = store.delete_traces(experiment_id, trace_ids=[trace_id])

        print(  # noqa: T201
            json.dumps(
                {
                    "archived": archived,
                    "spans_location": info.tags[TraceTagKey.SPANS_LOCATION],
                    "archive_suffix": f"/{experiment_id}/traces/{trace_id}/artifacts",
                    "archive_uri_has_suffix": archive_uri.endswith(
                        f"/{experiment_id}/traces/{trace_id}/artifacts"
                    ),
                    "payload_b64": base64.b64encode(payload).decode(),
                    "stored_content": stored_content,
                    "read_json": read_json,
                    "deleted": deleted,
                    "payload_exists_after_delete": payload_path.exists(),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
