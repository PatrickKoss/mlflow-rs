"""add query performance indexes

Revision ID: a3f8c21d9b47
Revises: b7e4c1a90f23

Create Date: 2026-07-17 10:00:00.000000

"""

from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "a3f8c21d9b47"
down_revision = "b7e4c1a90f23"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "index_runs_experiment_id_lifecycle_stage_start_time",
        "runs",
        ["experiment_id", "lifecycle_stage", "start_time"],
        unique=False,
    )
    op.create_index(
        "index_logged_models_experiment_id",
        "logged_models",
        ["experiment_id"],
        unique=False,
    )
    op.create_index(
        "index_inputs_source_id",
        "inputs",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        "index_model_versions_run_id",
        "model_versions",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "index_model_versions_current_stage",
        "model_versions",
        ["current_stage"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "index_model_versions_current_stage",
        table_name="model_versions",
    )
    op.drop_index(
        "index_model_versions_run_id",
        table_name="model_versions",
    )
    op.drop_index(
        "index_inputs_source_id",
        table_name="inputs",
    )
    # `logged_models.experiment_id` and `runs.experiment_id` both carry a foreign
    # key into `experiments`. On MySQL that column must always be backed by an
    # index, and MySQL consolidates the FK's implicit index into our explicit
    # index, so a plain `DROP INDEX` fails with errno 1553. `_drop_fk_backed_index`
    # drops and recreates the FK around the index drop on MySQL so MySQL can
    # regenerate its own implicit backing index; other dialects drop the index
    # directly.
    _drop_fk_backed_index("logged_models", "index_logged_models_experiment_id", "experiment_id")
    _drop_fk_backed_index(
        "runs", "index_runs_experiment_id_lifecycle_stage_start_time", "experiment_id"
    )


def _drop_fk_backed_index(table_name, index_name, fk_column):
    bind = op.get_bind()
    if bind.dialect.name != "mysql":
        op.drop_index(index_name, table_name=table_name)
        return

    fk = _find_single_column_foreign_key(bind, table_name, fk_column)
    if fk is None:
        op.drop_index(index_name, table_name=table_name)
        return

    op.drop_constraint(fk["name"], table_name, type_="foreignkey")
    op.drop_index(index_name, table_name=table_name)
    op.create_foreign_key(
        fk["name"],
        table_name,
        fk["referred_table"],
        [fk_column],
        fk["referred_columns"],
        ondelete=(fk.get("options") or {}).get("ondelete"),
        onupdate=(fk.get("options") or {}).get("onupdate"),
    )


def _find_single_column_foreign_key(bind, table_name, column_name):
    inspector = inspect(bind)
    return next(
        (
            fk
            for fk in inspector.get_foreign_keys(table_name)
            if fk.get("constrained_columns") == [column_name]
        ),
        None,
    )
