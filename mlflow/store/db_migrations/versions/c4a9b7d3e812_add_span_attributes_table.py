"""add span attributes table

Revision ID: c4a9b7d3e812
Revises: a3f8c21d9b47

Create Date: 2026-07-18 10:00:00.000000

"""

import json

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c4a9b7d3e812"
down_revision = "a3f8c21d9b47"
branch_labels = None
depends_on = None

_BACKFILL_SPAN_BATCH_SIZE = 1_000
_INSERT_ATTRIBUTE_BATCH_SIZE = 1_000
_MAX_ATTRIBUTE_KEY_LENGTH = 250
_MAX_ATTRIBUTE_VALUE_LENGTH = 500


def upgrade():
    op.create_table(
        "span_attributes",
        sa.Column("trace_id", sa.String(50), nullable=False),
        sa.Column("span_id", sa.String(50), nullable=False),
        sa.Column("key", sa.String(250), nullable=False),
        sa.Column("value", sa.String(500), nullable=False),
        sa.Column(
            "value_truncated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.PrimaryKeyConstraint("trace_id", "span_id", "key", name="span_attributes_pk"),
        sa.ForeignKeyConstraint(
            ["trace_id", "span_id"],
            ["spans.trace_id", "spans.span_id"],
            name="fk_span_attributes_span",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "index_span_attributes_key_value",
        "span_attributes",
        ["key", "value"],
        unique=False,
    )
    _backfill_span_attributes()


def downgrade():
    op.drop_index("index_span_attributes_key_value", table_name="span_attributes")
    op.drop_table("span_attributes")


def _backfill_span_attributes():
    bind = op.get_bind()
    spans = sa.table(
        "spans",
        sa.column("trace_id", sa.String(50)),
        sa.column("span_id", sa.String(50)),
        sa.column("content", sa.Text()),
    )
    span_attributes = sa.table(
        "span_attributes",
        sa.column("trace_id", sa.String(50)),
        sa.column("span_id", sa.String(50)),
        sa.column("key", sa.String(250)),
        sa.column("value", sa.String(500)),
        sa.column("value_truncated", sa.Boolean()),
    )

    last_trace_id = None
    last_span_id = None
    while True:
        stmt = (
            sa
            .select(spans.c.trace_id, spans.c.span_id, spans.c.content)
            .order_by(spans.c.trace_id, spans.c.span_id)
            .limit(_BACKFILL_SPAN_BATCH_SIZE)
        )
        if last_trace_id is not None:
            stmt = stmt.where(
                sa.or_(
                    spans.c.trace_id > last_trace_id,
                    sa.and_(
                        spans.c.trace_id == last_trace_id,
                        spans.c.span_id > last_span_id,
                    ),
                )
            )
        batch = bind.execute(stmt).fetchall()
        if not batch:
            break

        attribute_rows = []
        for row in batch:
            attribute_rows.extend(_extract_attribute_rows(row.trace_id, row.span_id, row.content))
        for offset in range(0, len(attribute_rows), _INSERT_ATTRIBUTE_BATCH_SIZE):
            op.bulk_insert(
                span_attributes,
                attribute_rows[offset : offset + _INSERT_ATTRIBUTE_BATCH_SIZE],
            )

        last_trace_id = batch[-1].trace_id
        last_span_id = batch[-1].span_id


def _extract_attribute_rows(trace_id, span_id, content):
    try:
        attributes = json.loads(content).get("attributes", {})
    except (AttributeError, TypeError, ValueError):
        return []
    if not isinstance(attributes, dict):
        return []

    rows = []
    for key, value in attributes.items():
        # Attribute keys are bounded rather than truncated: truncating keys can
        # alias two distinct attributes. Such unusual keys remain searchable by
        # the legacy content predicate.
        if not isinstance(key, str) or len(key) > _MAX_ATTRIBUTE_KEY_LENGTH:
            continue
        if not isinstance(value, str):
            continue
        rows.append({
            "trace_id": trace_id,
            "span_id": span_id,
            "key": key,
            "value": value[:_MAX_ATTRIBUTE_VALUE_LENGTH],
            "value_truncated": len(value) > _MAX_ATTRIBUTE_VALUE_LENGTH,
        })
    return rows
