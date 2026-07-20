from __future__ import annotations

import uuid

import pytest

import mlflow
from mlflow.genai.label_schemas import (
    InputPassFail,
    create_label_schema,
    delete_label_schema,
    get_label_schema,
    list_label_schemas,
    update_label_schema,
)
from mlflow.genai.review_queues import (
    create_review_queue,
    delete_review_queue,
    get_review_queue,
    list_review_queues,
    update_review_queue,
)
from mlflow.tracking import MlflowClient


@pytest.fixture
def tracking_client(db_uri):
    original = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(db_uri)
    client = MlflowClient(db_uri)
    experiment_id = client.create_experiment(f"t22-2-{uuid.uuid4().hex}")
    yield client, experiment_id
    mlflow.set_tracking_uri(original)


def test_evaluation_dataset_sdk_round_trip(tracking_client):
    client, experiment_id = tracking_client
    name = f"dataset-{uuid.uuid4().hex}"
    created = client.create_dataset(name, experiment_id=experiment_id, tags={"phase": "22.2"})

    assert client.get_dataset(created.dataset_id).name == name
    assert [dataset.dataset_id for dataset in client.search_datasets([experiment_id])] == [
        created.dataset_id
    ]

    client.delete_dataset(created.dataset_id)
    assert list(client.search_datasets([experiment_id])) == []


def test_label_schema_sdk_round_trip(tracking_client):
    _, experiment_id = tracking_client
    name = f"schema-{uuid.uuid4().hex}"
    created = create_label_schema(
        name,
        type="feedback",
        input=InputPassFail(positive_label="yes", negative_label="no"),
        experiment_id=experiment_id,
    )

    assert get_label_schema(schema_id=created.schema_id).name == name
    assert created.schema_id in {schema.schema_id for schema in list_label_schemas(experiment_id)}
    assert update_label_schema(created.schema_id, instruction="updated").instruction == "updated"

    delete_label_schema(schema_id=created.schema_id)
    assert created.schema_id not in {
        schema.schema_id for schema in list_label_schemas(experiment_id)
    }


def test_review_queue_sdk_round_trip(tracking_client):
    _, experiment_id = tracking_client
    name = f"queue-{uuid.uuid4().hex}"
    created = create_review_queue(
        name,
        queue_type="custom",
        users=["reviewer"],
        experiment_id=experiment_id,
    )

    assert get_review_queue(created.queue_id).name == name
    assert [queue.queue_id for queue in list_review_queues(experiment_id=experiment_id)] == [
        created.queue_id
    ]
    assert update_review_queue(created.queue_id, users=["owner"]).users == ["owner"]

    delete_review_queue(created.queue_id)
    assert list(list_review_queues(experiment_id=experiment_id)) == []
