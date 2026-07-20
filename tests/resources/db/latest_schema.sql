
CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL,
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
)


CREATE TABLE budget_policies (
	budget_policy_id VARCHAR(36) NOT NULL,
	budget_unit VARCHAR(32) NOT NULL,
	budget_amount FLOAT NOT NULL,
	duration_unit VARCHAR(32) NOT NULL,
	duration_value INTEGER NOT NULL,
	target_scope VARCHAR(32) NOT NULL,
	budget_action VARCHAR(32) NOT NULL,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT budget_policies_pk PRIMARY KEY (budget_policy_id)
)

CREATE INDEX idx_budget_policies_workspace ON budget_policies (workspace)

CREATE TABLE entity_associations (
	association_id VARCHAR(36) NOT NULL,
	source_type VARCHAR(36) NOT NULL,
	source_id VARCHAR(36) NOT NULL,
	destination_type VARCHAR(36) NOT NULL,
	destination_id VARCHAR(36) NOT NULL,
	created_time BIGINT,
	CONSTRAINT entity_associations_pk PRIMARY KEY (source_type, source_id, destination_type, destination_id)
)

CREATE INDEX index_entity_associations_association_id ON entity_associations (association_id)
CREATE INDEX index_entity_associations_reverse_lookup ON entity_associations (destination_type, destination_id, source_type, source_id)

CREATE TABLE evaluation_datasets (
	dataset_id VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	schema TEXT,
	profile TEXT,
	digest VARCHAR(64),
	created_time BIGINT,
	last_update_time BIGINT,
	created_by VARCHAR(255),
	last_updated_by VARCHAR(255),
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT evaluation_datasets_pk PRIMARY KEY (dataset_id)
)

CREATE INDEX idx_evaluation_datasets_workspace ON evaluation_datasets (workspace)
CREATE INDEX index_evaluation_datasets_created_time ON evaluation_datasets (created_time)
CREATE INDEX index_evaluation_datasets_name ON evaluation_datasets (name)

CREATE TABLE experiments (
	experiment_id INTEGER NOT NULL,
	name VARCHAR(256) NOT NULL,
	artifact_location VARCHAR(256),
	lifecycle_stage VARCHAR(32),
	creation_time BIGINT,
	last_update_time BIGINT,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT experiment_pk PRIMARY KEY (experiment_id),
	CONSTRAINT uq_experiments_workspace_name UNIQUE (workspace, name),
	CONSTRAINT experiments_lifecycle_stage CHECK (lifecycle_stage IN ('active', 'deleted'))
)

CREATE INDEX idx_experiments_workspace ON experiments (workspace)
CREATE INDEX idx_experiments_workspace_creation_time ON experiments (workspace, creation_time)

CREATE TABLE input_tags (
	input_uuid VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	value VARCHAR(500) NOT NULL,
	CONSTRAINT input_tags_pk PRIMARY KEY (input_uuid, name)
)


CREATE TABLE inputs (
	input_uuid VARCHAR(36) NOT NULL,
	source_type VARCHAR(36) NOT NULL,
	source_id VARCHAR(36) NOT NULL,
	destination_type VARCHAR(36) NOT NULL,
	destination_id VARCHAR(36) NOT NULL,
	step BIGINT DEFAULT '0' NOT NULL,
	CONSTRAINT inputs_pk PRIMARY KEY (source_type, source_id, destination_type, destination_id)
)

CREATE INDEX index_inputs_destination_type_destination_id_source_type ON inputs (destination_type, destination_id, source_type)
CREATE INDEX index_inputs_input_uuid ON inputs (input_uuid)
CREATE INDEX index_inputs_source_id ON inputs (source_id)

CREATE TABLE jobs (
	id VARCHAR(36) NOT NULL,
	creation_time BIGINT NOT NULL,
	job_name VARCHAR(500) NOT NULL,
	params TEXT NOT NULL,
	timeout FLOAT,
	status INTEGER NOT NULL,
	result TEXT,
	retry_count INTEGER NOT NULL,
	last_update_time BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	status_details JSON,
	CONSTRAINT jobs_pk PRIMARY KEY (id)
)

CREATE INDEX index_jobs_name_status_creation_time ON jobs (job_name, workspace, status, creation_time)

CREATE TABLE mcp_servers (
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	name VARCHAR(256) NOT NULL,
	display_name VARCHAR(256),
	description TEXT,
	icons JSON,
	created_by VARCHAR(256),
	last_updated_by VARCHAR(256),
	created_at BIGINT NOT NULL,
	last_updated_at BIGINT NOT NULL,
	CONSTRAINT mcp_servers_pk PRIMARY KEY (workspace, name)
)


CREATE TABLE registered_models (
	name VARCHAR(256) NOT NULL,
	creation_time BIGINT,
	last_updated_time BIGINT,
	description VARCHAR(5000),
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT registered_model_pk PRIMARY KEY (workspace, name)
)

CREATE INDEX idx_registered_models_workspace ON registered_models (workspace)

CREATE TABLE secrets (
	secret_id VARCHAR(36) NOT NULL,
	secret_name VARCHAR(255) NOT NULL,
	encrypted_value BLOB NOT NULL,
	wrapped_dek BLOB NOT NULL,
	kek_version INTEGER NOT NULL,
	masked_value VARCHAR(500) NOT NULL,
	provider VARCHAR(64),
	auth_config TEXT,
	description TEXT,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT secrets_pk PRIMARY KEY (secret_id),
	CONSTRAINT uq_secrets_workspace_secret_name UNIQUE (workspace, secret_name)
)

CREATE INDEX idx_secrets_workspace ON secrets (workspace)

CREATE TABLE webhooks (
	webhook_id VARCHAR(256) NOT NULL,
	name VARCHAR(256) NOT NULL,
	description VARCHAR(1000),
	url VARCHAR(500) NOT NULL,
	status VARCHAR(20) DEFAULT 'ACTIVE' NOT NULL,
	secret VARCHAR(1000),
	creation_timestamp BIGINT,
	last_updated_timestamp BIGINT,
	deleted_timestamp BIGINT,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT webhook_pk PRIMARY KEY (webhook_id)
)

CREATE INDEX idx_webhooks_name ON webhooks (name)
CREATE INDEX idx_webhooks_status ON webhooks (status)
CREATE INDEX idx_webhooks_workspace ON webhooks (workspace)

CREATE TABLE workspaces (
	name VARCHAR(63) NOT NULL,
	description TEXT,
	default_artifact_root TEXT,
	trace_archival_location TEXT,
	trace_archival_retention VARCHAR(32),
	CONSTRAINT workspaces_pk PRIMARY KEY (name)
)


CREATE TABLE datasets (
	dataset_uuid VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(500) NOT NULL,
	digest VARCHAR(36) NOT NULL,
	dataset_source_type VARCHAR(36) NOT NULL,
	dataset_source TEXT NOT NULL,
	dataset_schema TEXT,
	dataset_profile TEXT,
	CONSTRAINT dataset_pk PRIMARY KEY (experiment_id, name, digest),
	CONSTRAINT fk_datasets_experiment_id_experiments FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE
)

CREATE INDEX index_datasets_dataset_uuid ON datasets (dataset_uuid)
CREATE INDEX index_datasets_experiment_id_dataset_source_type ON datasets (experiment_id, dataset_source_type)

CREATE TABLE endpoints (
	endpoint_id VARCHAR(36) NOT NULL,
	name VARCHAR(255),
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	routing_strategy VARCHAR(64),
	fallback_config_json TEXT,
	experiment_id INTEGER,
	usage_tracking BOOLEAN DEFAULT '0' NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT endpoints_pk PRIMARY KEY (endpoint_id),
	CONSTRAINT fk_endpoints_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE SET NULL,
	CONSTRAINT uq_endpoints_workspace_name UNIQUE (workspace, name)
)

CREATE INDEX idx_endpoints_workspace ON endpoints (workspace)

CREATE TABLE evaluation_dataset_records (
	dataset_record_id VARCHAR(36) NOT NULL,
	dataset_id VARCHAR(36) NOT NULL,
	inputs JSON NOT NULL,
	expectations JSON,
	tags JSON,
	source JSON,
	source_id VARCHAR(36),
	source_type VARCHAR(255),
	created_time BIGINT,
	last_update_time BIGINT,
	created_by VARCHAR(255),
	last_updated_by VARCHAR(255),
	input_hash VARCHAR(64) NOT NULL,
	outputs JSON,
	CONSTRAINT evaluation_dataset_records_pk PRIMARY KEY (dataset_record_id),
	CONSTRAINT fk_evaluation_dataset_records_dataset_id FOREIGN KEY(dataset_id) REFERENCES evaluation_datasets (dataset_id) ON DELETE CASCADE,
	CONSTRAINT unique_dataset_input UNIQUE (dataset_id, input_hash)
)

CREATE INDEX index_evaluation_dataset_records_dataset_id ON evaluation_dataset_records (dataset_id)

CREATE TABLE evaluation_dataset_tags (
	dataset_id VARCHAR(36) NOT NULL,
	key VARCHAR(255) NOT NULL,
	value VARCHAR(5000),
	CONSTRAINT evaluation_dataset_tags_pk PRIMARY KEY (dataset_id, key),
	CONSTRAINT fk_evaluation_dataset_tags_dataset_id FOREIGN KEY(dataset_id) REFERENCES evaluation_datasets (dataset_id) ON DELETE CASCADE
)

CREATE INDEX index_evaluation_dataset_tags_dataset_id ON evaluation_dataset_tags (dataset_id)

CREATE TABLE experiment_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	experiment_id INTEGER NOT NULL,
	CONSTRAINT experiment_tag_pk PRIMARY KEY (key, experiment_id),
	FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)


CREATE TABLE label_schemas (
	schema_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(250) NOT NULL,
	type VARCHAR(16) NOT NULL,
	instruction TEXT,
	enable_comment BOOLEAN DEFAULT '0' NOT NULL,
	input_type VARCHAR(32) NOT NULL,
	input_config TEXT NOT NULL,
	created_by VARCHAR(255),
	created_time BIGINT NOT NULL,
	last_update_time BIGINT NOT NULL,
	is_default BOOLEAN DEFAULT 0 NOT NULL,
	CONSTRAINT label_schemas_pk PRIMARY KEY (schema_id),
	CONSTRAINT fk_label_schemas_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE,
	CONSTRAINT uq_label_schemas_exp_name UNIQUE (experiment_id, name)
)

CREATE INDEX index_label_schemas_experiment_id ON label_schemas (experiment_id)

CREATE TABLE logged_models (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(500) NOT NULL,
	artifact_location VARCHAR(1000) NOT NULL,
	creation_timestamp_ms BIGINT NOT NULL,
	last_updated_timestamp_ms BIGINT NOT NULL,
	status INTEGER NOT NULL,
	lifecycle_stage VARCHAR(32),
	model_type VARCHAR(500),
	source_run_id VARCHAR(32),
	status_message VARCHAR(1000),
	CONSTRAINT logged_models_pk PRIMARY KEY (model_id),
	CONSTRAINT fk_logged_models_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE,
	CONSTRAINT logged_models_lifecycle_stage_check CHECK (lifecycle_stage IN ('active', 'deleted'))
)

CREATE INDEX index_logged_models_experiment_id ON logged_models (experiment_id)

CREATE TABLE mcp_access_endpoints (
	id VARCHAR(36) NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	server_name VARCHAR(256) NOT NULL,
	server_version VARCHAR(128),
	server_alias VARCHAR(256),
	url VARCHAR(2048) NOT NULL,
	transport_type VARCHAR(32) DEFAULT 'streamable-http' NOT NULL,
	created_by VARCHAR(256),
	last_updated_by VARCHAR(256),
	created_at BIGINT NOT NULL,
	last_updated_at BIGINT NOT NULL,
	CONSTRAINT mcp_access_endpoints_pk PRIMARY KEY (id),
	CONSTRAINT mcp_access_endpoints_server_fkey FOREIGN KEY(workspace, server_name) REFERENCES mcp_servers (workspace, name) ON DELETE CASCADE ON UPDATE CASCADE
)


CREATE TABLE mcp_server_aliases (
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	name VARCHAR(256) NOT NULL,
	alias VARCHAR(256) NOT NULL,
	version VARCHAR(128) NOT NULL,
	CONSTRAINT mcp_server_aliases_pk PRIMARY KEY (workspace, name, alias),
	CONSTRAINT mcp_server_aliases_server_fkey FOREIGN KEY(workspace, name) REFERENCES mcp_servers (workspace, name) ON DELETE CASCADE ON UPDATE CASCADE
)


CREATE TABLE mcp_server_tags (
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	name VARCHAR(256) NOT NULL,
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	CONSTRAINT mcp_server_tags_pk PRIMARY KEY (workspace, name, key),
	CONSTRAINT mcp_server_tags_server_fkey FOREIGN KEY(workspace, name) REFERENCES mcp_servers (workspace, name) ON DELETE CASCADE ON UPDATE CASCADE
)


CREATE TABLE mcp_server_versions (
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	name VARCHAR(256) NOT NULL,
	version VARCHAR(128) NOT NULL,
	version_major INTEGER NOT NULL,
	version_minor INTEGER NOT NULL,
	version_patch INTEGER NOT NULL,
	version_prerelease_sort_key VARCHAR(512) NOT NULL,
	server_json JSON NOT NULL,
	display_name VARCHAR(256),
	status VARCHAR(20) DEFAULT 'draft' NOT NULL,
	tools JSON,
	source VARCHAR(512),
	created_by VARCHAR(256),
	last_updated_by VARCHAR(256),
	created_at BIGINT NOT NULL,
	last_updated_at BIGINT NOT NULL,
	CONSTRAINT mcp_server_versions_pk PRIMARY KEY (workspace, name, version),
	CONSTRAINT mcp_server_versions_server_fkey FOREIGN KEY(workspace, name) REFERENCES mcp_servers (workspace, name) ON DELETE CASCADE ON UPDATE CASCADE
)


CREATE TABLE model_definitions (
	model_definition_id VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	secret_id VARCHAR(36),
	provider VARCHAR(64) NOT NULL,
	model_name VARCHAR(256) NOT NULL,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT model_definitions_pk PRIMARY KEY (model_definition_id),
	CONSTRAINT fk_model_definitions_secret_id FOREIGN KEY(secret_id) REFERENCES secrets (secret_id) ON DELETE SET NULL,
	CONSTRAINT uq_model_definitions_workspace_name UNIQUE (workspace, name)
)

CREATE INDEX idx_model_definitions_workspace ON model_definitions (workspace)
CREATE INDEX index_model_definitions_provider ON model_definitions (provider)
CREATE INDEX index_model_definitions_secret_id ON model_definitions (secret_id)

CREATE TABLE model_versions (
	name VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	creation_time BIGINT,
	last_updated_time BIGINT,
	description VARCHAR(5000),
	user_id VARCHAR(256),
	current_stage VARCHAR(20),
	source VARCHAR(500),
	run_id VARCHAR(32),
	status VARCHAR(20),
	status_message VARCHAR(500),
	run_link VARCHAR(500),
	storage_location VARCHAR(500),
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT model_version_pk PRIMARY KEY (workspace, name, version),
	CONSTRAINT fk_model_versions_registered_models FOREIGN KEY(workspace, name) REFERENCES registered_models (workspace, name) ON UPDATE CASCADE
)

CREATE INDEX index_model_versions_current_stage ON model_versions (current_stage)
CREATE INDEX index_model_versions_run_id ON model_versions (run_id)

CREATE TABLE registered_model_aliases (
	alias VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	name VARCHAR(256) NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT registered_model_alias_pk PRIMARY KEY (workspace, name, alias),
	CONSTRAINT fk_registered_model_aliases_registered_models FOREIGN KEY(workspace, name) REFERENCES registered_models (workspace, name) ON DELETE CASCADE ON UPDATE CASCADE
)


CREATE TABLE registered_model_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	name VARCHAR(256) NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT registered_model_tag_pk PRIMARY KEY (workspace, key, name),
	CONSTRAINT fk_registered_model_tags_registered_models FOREIGN KEY(workspace, name) REFERENCES registered_models (workspace, name) ON UPDATE CASCADE
)

CREATE INDEX idx_registered_model_tags_workspace_name ON registered_model_tags (workspace, name)

CREATE TABLE review_queues (
	queue_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(250) NOT NULL,
	queue_type VARCHAR(16) NOT NULL,
	created_by VARCHAR(255),
	creation_time_ms BIGINT NOT NULL,
	last_update_time_ms BIGINT NOT NULL,
	name_key VARCHAR(250) NOT NULL,
	CONSTRAINT review_queues_pk PRIMARY KEY (queue_id),
	CONSTRAINT fk_review_queues_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE,
	CONSTRAINT uq_review_queues_experiment_name_key UNIQUE (experiment_id, name_key)
)

CREATE INDEX index_review_queues_experiment_id ON review_queues (experiment_id)

CREATE TABLE runs (
	run_uuid VARCHAR(32) NOT NULL,
	name VARCHAR(250),
	source_type VARCHAR(20),
	source_name VARCHAR(500),
	entry_point_name VARCHAR(50),
	user_id VARCHAR(256),
	status VARCHAR(9),
	start_time BIGINT,
	end_time BIGINT,
	source_version VARCHAR(50),
	lifecycle_stage VARCHAR(20),
	artifact_uri VARCHAR(200),
	experiment_id INTEGER,
	deleted_time BIGINT,
	CONSTRAINT run_pk PRIMARY KEY (run_uuid),
	FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT runs_lifecycle_stage CHECK (lifecycle_stage IN ('active', 'deleted')),
	CONSTRAINT source_type CHECK (source_type IN ('NOTEBOOK', 'JOB', 'LOCAL', 'UNKNOWN', 'PROJECT')),
	CHECK (status IN ('SCHEDULED', 'FAILED', 'FINISHED', 'RUNNING', 'KILLED'))
)

CREATE INDEX index_runs_experiment_id_lifecycle_stage_start_time ON runs (experiment_id, lifecycle_stage, start_time)

CREATE TABLE scorers (
	experiment_id INTEGER NOT NULL,
	scorer_name VARCHAR(256) NOT NULL,
	scorer_id VARCHAR(36) NOT NULL,
	CONSTRAINT scorer_pk PRIMARY KEY (scorer_id),
	CONSTRAINT fk_scorers_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE
)

CREATE UNIQUE INDEX index_scorers_experiment_id_scorer_name ON scorers (experiment_id, scorer_name)

CREATE TABLE trace_info (
	request_id VARCHAR(50) NOT NULL,
	experiment_id INTEGER NOT NULL,
	timestamp_ms BIGINT NOT NULL,
	execution_time_ms BIGINT,
	status VARCHAR(50) NOT NULL,
	client_request_id VARCHAR(50),
	request_preview VARCHAR(1000),
	response_preview VARCHAR(1000),
	db_payload_generation INTEGER DEFAULT '0' NOT NULL,
	CONSTRAINT trace_info_pk PRIMARY KEY (request_id),
	CONSTRAINT fk_trace_info_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)

CREATE INDEX index_trace_info_experiment_id_timestamp_ms ON trace_info (experiment_id, timestamp_ms)

CREATE TABLE webhook_events (
	webhook_id VARCHAR(256) NOT NULL,
	entity VARCHAR(50) NOT NULL,
	action VARCHAR(50) NOT NULL,
	CONSTRAINT webhook_event_pk PRIMARY KEY (webhook_id, entity, action),
	FOREIGN KEY(webhook_id) REFERENCES webhooks (webhook_id) ON DELETE CASCADE
)

CREATE INDEX idx_webhook_events_action ON webhook_events (action)
CREATE INDEX idx_webhook_events_entity ON webhook_events (entity)
CREATE INDEX idx_webhook_events_entity_action ON webhook_events (entity, action)

CREATE TABLE assessments (
	assessment_id VARCHAR(50) NOT NULL,
	trace_id VARCHAR(50) NOT NULL,
	name VARCHAR(250) NOT NULL,
	assessment_type VARCHAR(20) NOT NULL,
	value TEXT NOT NULL,
	error TEXT,
	created_timestamp BIGINT NOT NULL,
	last_updated_timestamp BIGINT NOT NULL,
	source_type VARCHAR(50) NOT NULL,
	source_id VARCHAR(250),
	run_id VARCHAR(32),
	span_id VARCHAR(50),
	rationale TEXT,
	overrides VARCHAR(50),
	valid BOOLEAN NOT NULL,
	assessment_metadata TEXT,
	CONSTRAINT assessments_pk PRIMARY KEY (assessment_id),
	CONSTRAINT fk_assessments_trace_id FOREIGN KEY(trace_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
)

CREATE INDEX index_assessments_assessment_type ON assessments (assessment_type)
CREATE INDEX index_assessments_last_updated_timestamp ON assessments (last_updated_timestamp)
CREATE INDEX index_assessments_run_id_created_timestamp ON assessments (run_id, created_timestamp)
CREATE INDEX index_assessments_trace_id_created_timestamp ON assessments (trace_id, created_timestamp)

CREATE TABLE endpoint_bindings (
	endpoint_id VARCHAR(36) NOT NULL,
	resource_type VARCHAR(50) NOT NULL,
	resource_id VARCHAR(255) NOT NULL,
	created_at BIGINT NOT NULL,
	created_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	display_name VARCHAR(255),
	CONSTRAINT endpoint_bindings_pk PRIMARY KEY (endpoint_id, resource_type, resource_id),
	CONSTRAINT fk_endpoint_bindings_endpoint_id FOREIGN KEY(endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE CASCADE
)


CREATE TABLE endpoint_model_mappings (
	mapping_id VARCHAR(36) NOT NULL,
	endpoint_id VARCHAR(36) NOT NULL,
	model_definition_id VARCHAR(36) NOT NULL,
	weight FLOAT NOT NULL,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	linkage_type VARCHAR(64) DEFAULT 'PRIMARY' NOT NULL,
	fallback_order INTEGER,
	CONSTRAINT endpoint_model_mappings_pk PRIMARY KEY (mapping_id),
	CONSTRAINT fk_endpoint_model_mappings_endpoint_id FOREIGN KEY(endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE CASCADE,
	CONSTRAINT fk_endpoint_model_mappings_model_definition_id FOREIGN KEY(model_definition_id) REFERENCES model_definitions (model_definition_id)
)

CREATE INDEX index_endpoint_model_mappings_endpoint_id ON endpoint_model_mappings (endpoint_id)
CREATE INDEX index_endpoint_model_mappings_model_definition_id ON endpoint_model_mappings (model_definition_id)
CREATE UNIQUE INDEX unique_endpoint_model_linkage_mapping ON endpoint_model_mappings (endpoint_id, model_definition_id, linkage_type)

CREATE TABLE endpoint_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	endpoint_id VARCHAR(36) NOT NULL,
	CONSTRAINT endpoint_tag_pk PRIMARY KEY (key, endpoint_id),
	CONSTRAINT fk_endpoint_tags_endpoint_id FOREIGN KEY(endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE CASCADE
)

CREATE INDEX index_endpoint_tags_endpoint_id ON endpoint_tags (endpoint_id)

CREATE TABLE issues (
	issue_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(250) NOT NULL,
	description TEXT NOT NULL,
	status VARCHAR(50) NOT NULL,
	severity VARCHAR(50),
	root_causes TEXT,
	source_run_id VARCHAR(32),
	categories TEXT,
	created_timestamp BIGINT NOT NULL,
	last_updated_timestamp BIGINT NOT NULL,
	created_by VARCHAR(255),
	CONSTRAINT issues_pk PRIMARY KEY (issue_id),
	CONSTRAINT fk_issues_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE,
	CONSTRAINT fk_issues_source_run_id FOREIGN KEY(source_run_id) REFERENCES runs (run_uuid) ON DELETE SET NULL
)

CREATE INDEX index_issues_experiment_id ON issues (experiment_id)
CREATE INDEX index_issues_source_run_id ON issues (source_run_id)
CREATE INDEX index_issues_status ON issues (status)

CREATE TABLE latest_metrics (
	key VARCHAR(250) NOT NULL,
	value FLOAT NOT NULL,
	timestamp BIGINT,
	step BIGINT NOT NULL,
	is_nan BOOLEAN NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT latest_metric_pk PRIMARY KEY (key, run_uuid),
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid),
	CHECK (is_nan IN (0, 1))
)

CREATE INDEX index_latest_metrics_run_uuid ON latest_metrics (run_uuid)

CREATE TABLE logged_model_metrics (
	model_id VARCHAR(36) NOT NULL,
	metric_name VARCHAR(500) NOT NULL,
	metric_timestamp_ms BIGINT NOT NULL,
	metric_step BIGINT NOT NULL,
	metric_value FLOAT,
	experiment_id INTEGER NOT NULL,
	run_id VARCHAR(32) NOT NULL,
	dataset_uuid VARCHAR(36),
	dataset_name VARCHAR(500),
	dataset_digest VARCHAR(36),
	CONSTRAINT logged_model_metrics_pk PRIMARY KEY (model_id, metric_name, metric_timestamp_ms, metric_step, run_id),
	CONSTRAINT fk_logged_model_metrics_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_metrics_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE,
	CONSTRAINT fk_logged_model_metrics_run_id FOREIGN KEY(run_id) REFERENCES runs (run_uuid) ON DELETE CASCADE
)

CREATE INDEX index_logged_model_metrics_model_id ON logged_model_metrics (model_id)

CREATE TABLE logged_model_params (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	param_key VARCHAR(255) NOT NULL,
	param_value TEXT NOT NULL,
	CONSTRAINT logged_model_params_pk PRIMARY KEY (model_id, param_key),
	CONSTRAINT fk_logged_model_params_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_params_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE
)


CREATE TABLE logged_model_tags (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	tag_key VARCHAR(255) NOT NULL,
	tag_value TEXT NOT NULL,
	CONSTRAINT logged_model_tags_pk PRIMARY KEY (model_id, tag_key),
	CONSTRAINT fk_logged_model_tags_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_tags_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE
)


CREATE TABLE mcp_server_version_tags (
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	name VARCHAR(256) NOT NULL,
	version VARCHAR(128) NOT NULL,
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	CONSTRAINT mcp_server_version_tags_pk PRIMARY KEY (workspace, name, version, key),
	CONSTRAINT mcp_server_version_tags_version_fkey FOREIGN KEY(workspace, name, version) REFERENCES mcp_server_versions (workspace, name, version) ON DELETE CASCADE ON UPDATE CASCADE
)


CREATE TABLE metrics (
	key VARCHAR(250) NOT NULL,
	value FLOAT NOT NULL,
	timestamp BIGINT NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	step BIGINT DEFAULT '0' NOT NULL,
	is_nan BOOLEAN DEFAULT '0' NOT NULL,
	CONSTRAINT metric_pk PRIMARY KEY (key, timestamp, step, run_uuid, value, is_nan),
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid),
	CHECK (is_nan IN (0, 1))
)

CREATE INDEX index_metrics_run_uuid ON metrics (run_uuid)
CREATE INDEX index_metrics_run_uuid_key_step ON metrics (run_uuid, key, step)

CREATE TABLE model_version_tags (
	key VARCHAR(250) NOT NULL,
	value TEXT,
	name VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT model_version_tag_pk PRIMARY KEY (workspace, key, name, version),
	CONSTRAINT fk_model_version_tags_model_versions FOREIGN KEY(workspace, name, version) REFERENCES model_versions (workspace, name, version) ON UPDATE CASCADE
)

CREATE INDEX idx_model_version_tags_workspace_name_version ON model_version_tags (workspace, name, version)

CREATE TABLE online_scoring_configs (
	online_scoring_config_id VARCHAR(36) NOT NULL,
	scorer_id VARCHAR(36) NOT NULL,
	sample_rate FLOAT NOT NULL,
	experiment_id INTEGER NOT NULL,
	filter_string TEXT,
	CONSTRAINT online_scoring_config_pk PRIMARY KEY (online_scoring_config_id),
	CONSTRAINT fk_online_scoring_configs_scorer_id FOREIGN KEY(scorer_id) REFERENCES scorers (scorer_id) ON DELETE CASCADE,
	CONSTRAINT fk_online_scoring_configs_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)


CREATE TABLE params (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000) NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT param_pk PRIMARY KEY (key, run_uuid),
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)

CREATE INDEX index_params_run_uuid ON params (run_uuid)

CREATE TABLE review_queue_items (
	queue_id VARCHAR(36) NOT NULL,
	item_type VARCHAR(16) NOT NULL,
	item_id VARCHAR(50) NOT NULL,
	status VARCHAR(16) NOT NULL,
	completed_by VARCHAR(250),
	completed_time_ms BIGINT,
	creation_time_ms BIGINT NOT NULL,
	last_update_time_ms BIGINT NOT NULL,
	CONSTRAINT review_queue_items_pk PRIMARY KEY (queue_id, item_id),
	CONSTRAINT fk_review_queue_items_queue_id FOREIGN KEY(queue_id) REFERENCES review_queues (queue_id) ON DELETE CASCADE
)

CREATE INDEX index_review_queue_items_item_id ON review_queue_items (item_id)
CREATE INDEX index_review_queue_items_queue_id_status ON review_queue_items (queue_id, status)

CREATE TABLE review_queue_label_schemas (
	queue_id VARCHAR(36) NOT NULL,
	schema_id VARCHAR(36) NOT NULL,
	CONSTRAINT review_queue_label_schemas_pk PRIMARY KEY (queue_id, schema_id),
	CONSTRAINT fk_review_queue_label_schemas_queue_id FOREIGN KEY(queue_id) REFERENCES review_queues (queue_id) ON DELETE CASCADE
)

CREATE INDEX index_review_queue_label_schemas_schema_id ON review_queue_label_schemas (schema_id)

CREATE TABLE review_queue_users (
	queue_id VARCHAR(36) NOT NULL,
	user_id VARCHAR(250) NOT NULL,
	CONSTRAINT review_queue_users_pk PRIMARY KEY (queue_id, user_id),
	CONSTRAINT fk_review_queue_users_queue_id FOREIGN KEY(queue_id) REFERENCES review_queues (queue_id) ON DELETE CASCADE
)

CREATE INDEX index_review_queue_users_user_id ON review_queue_users (user_id)

CREATE TABLE scorer_versions (
	scorer_id VARCHAR(36) NOT NULL,
	scorer_version INTEGER NOT NULL,
	serialized_scorer TEXT NOT NULL,
	creation_time BIGINT,
	CONSTRAINT scorer_version_pk PRIMARY KEY (scorer_id, scorer_version),
	CONSTRAINT fk_scorer_versions_scorer_id FOREIGN KEY(scorer_id) REFERENCES scorers (scorer_id) ON DELETE CASCADE
)

CREATE INDEX index_scorer_versions_scorer_id ON scorer_versions (scorer_id)

CREATE TABLE spans (
	trace_id VARCHAR(50) NOT NULL,
	experiment_id INTEGER NOT NULL,
	span_id VARCHAR(50) NOT NULL,
	parent_span_id VARCHAR(50),
	name TEXT,
	type VARCHAR(500),
	status VARCHAR(50) NOT NULL,
	start_time_unix_nano BIGINT NOT NULL,
	end_time_unix_nano BIGINT,
	duration_ns BIGINT GENERATED ALWAYS AS (end_time_unix_nano - start_time_unix_nano) STORED,
	content TEXT NOT NULL,
	dimension_attributes JSON,
	CONSTRAINT spans_pk PRIMARY KEY (trace_id, span_id),
	CONSTRAINT fk_spans_trace_id FOREIGN KEY(trace_id) REFERENCES trace_info (request_id) ON DELETE CASCADE,
	CONSTRAINT fk_spans_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)

CREATE INDEX index_spans_experiment_id ON spans (experiment_id)
CREATE INDEX index_spans_experiment_id_duration ON spans (experiment_id, duration_ns)
CREATE INDEX index_spans_experiment_id_status_type ON spans (experiment_id, status, type)
CREATE INDEX index_spans_experiment_id_type_status ON spans (experiment_id, type, status)

CREATE TABLE tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT tag_pk PRIMARY KEY (key, run_uuid),
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)

CREATE INDEX index_tags_run_uuid ON tags (run_uuid)

CREATE TABLE trace_metrics (
	request_id VARCHAR(50) NOT NULL,
	key VARCHAR(250) NOT NULL,
	value FLOAT,
	CONSTRAINT trace_metrics_pk PRIMARY KEY (request_id, key),
	CONSTRAINT fk_trace_metrics_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
)

CREATE INDEX index_trace_metrics_request_id ON trace_metrics (request_id)

CREATE TABLE trace_request_metadata (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	request_id VARCHAR(50) NOT NULL,
	CONSTRAINT trace_request_metadata_pk PRIMARY KEY (key, request_id),
	CONSTRAINT fk_trace_request_metadata_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
)

CREATE INDEX index_trace_request_metadata_request_id ON trace_request_metadata (request_id)

CREATE TABLE trace_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	request_id VARCHAR(50) NOT NULL,
	CONSTRAINT trace_tag_pk PRIMARY KEY (key, request_id),
	CONSTRAINT fk_trace_tags_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
)

CREATE INDEX index_trace_tags_request_id ON trace_tags (request_id)

CREATE TABLE guardrails (
	guardrail_id VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	scorer_id VARCHAR(36) NOT NULL,
	scorer_version INTEGER NOT NULL,
	stage VARCHAR(32) NOT NULL,
	action VARCHAR(32) NOT NULL,
	action_endpoint_id VARCHAR(36),
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	last_updated_by VARCHAR(255),
	last_updated_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT guardrails_pk PRIMARY KEY (guardrail_id),
	CONSTRAINT fk_guardrails_scorer_version FOREIGN KEY(scorer_id, scorer_version) REFERENCES scorer_versions (scorer_id, scorer_version),
	CONSTRAINT fk_guardrails_action_endpoint_id FOREIGN KEY(action_endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE SET NULL
)

CREATE INDEX idx_guardrails_scorer ON guardrails (scorer_id, scorer_version)
CREATE INDEX idx_guardrails_workspace ON guardrails (workspace)

CREATE TABLE span_attributes (
	trace_id VARCHAR(50) NOT NULL,
	span_id VARCHAR(50) NOT NULL,
	key VARCHAR(250) NOT NULL,
	value VARCHAR(500) NOT NULL,
	value_truncated BOOLEAN DEFAULT 0 NOT NULL,
	CONSTRAINT span_attributes_pk PRIMARY KEY (trace_id, span_id, key),
	CONSTRAINT fk_span_attributes_span FOREIGN KEY(trace_id, span_id) REFERENCES spans (trace_id, span_id) ON DELETE CASCADE
)

CREATE INDEX index_span_attributes_key_value ON span_attributes (key, value)

CREATE TABLE span_metrics (
	trace_id VARCHAR(50) NOT NULL,
	span_id VARCHAR(50) NOT NULL,
	key VARCHAR(250) NOT NULL,
	value FLOAT,
	CONSTRAINT span_metrics_pk PRIMARY KEY (trace_id, span_id, key),
	CONSTRAINT fk_span_metrics_span FOREIGN KEY(trace_id, span_id) REFERENCES spans (trace_id, span_id) ON DELETE CASCADE
)

CREATE INDEX index_span_metrics_trace_id_span_id ON span_metrics (trace_id, span_id)

CREATE TABLE guardrail_configs (
	endpoint_id VARCHAR(36) NOT NULL,
	guardrail_id VARCHAR(36) NOT NULL,
	execution_order INTEGER,
	created_by VARCHAR(255),
	created_at BIGINT NOT NULL,
	workspace VARCHAR(63) DEFAULT 'default' NOT NULL,
	CONSTRAINT guardrail_configs_pk PRIMARY KEY (endpoint_id, guardrail_id),
	CONSTRAINT fk_guardrail_configs_endpoint_id FOREIGN KEY(endpoint_id) REFERENCES endpoints (endpoint_id) ON DELETE CASCADE,
	CONSTRAINT fk_guardrail_configs_guardrail_id FOREIGN KEY(guardrail_id) REFERENCES guardrails (guardrail_id) ON DELETE CASCADE
)

CREATE INDEX idx_guardrail_configs_endpoint_id ON guardrail_configs (endpoint_id)
CREATE INDEX idx_guardrail_configs_guardrail_id ON guardrail_configs (guardrail_id)
