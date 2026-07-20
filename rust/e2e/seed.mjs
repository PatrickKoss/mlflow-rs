import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const baseURL = (process.env.MLFLOW_E2E_BASE_URL ?? "http://127.0.0.1").replace(/\/$/, "");

async function request(method, route, body, expected = 200, headers = {}) {
  const response = await fetch(`${baseURL}${route}`, {
    method,
    headers: {
      ...(body === undefined ? {} : { "content-type": "application/json" }),
      ...headers,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  const backend = response.headers.get("x-mlflow-backend");
  if (backend !== "rust") {
    throw new Error(`${method} ${route}: expected Rust attribution, got ${backend ?? "<missing>"}`);
  }
  if (response.status !== expected) {
    throw new Error(`${method} ${route}: expected ${expected}, got ${response.status}: ${text}`);
  }
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function uploadArtifact(artifactPath, contents) {
  const response = await fetch(
    `${baseURL}/api/2.0/mlflow-artifacts/artifacts/${artifactPath.replace(/^\/+/, "")}`,
    {
      method: "PUT",
      headers: { "content-type": "application/octet-stream" },
      body: contents,
    },
  );
  const text = await response.text();
  if (response.headers.get("x-mlflow-backend") !== "rust" || response.status !== 200) {
    throw new Error(`PUT artifact ${artifactPath}: ${response.status}: ${text}`);
  }
}

function artifactProxyPath(uri, child = "") {
  const root = uri
    .replace(/^mlflow-artifacts:\/\//, "")
    .replace(/^mlflow-artifacts:\//, "")
    .replace(/^file:\/\//, "")
    .replace(/^\/mlartifacts\//, "")
    .replace(/^\/+/, "")
    .replace(/\/+$/, "");
  return [root, child].filter(Boolean).join("/");
}

async function waitForHealth() {
  for (let attempt = 0; attempt < 90; attempt += 1) {
    try {
      await request("GET", "/health");
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 1_000));
    }
  }
  throw new Error(`${baseURL}/health did not become ready`);
}

function otlpTrace(traceHex, rootSpanHex, childSpanHex, index) {
  const start = BigInt("1784563200000000000") + BigInt(index) * BigInt("1000000000");
  const traceId = `tr-${traceHex}`;
  const attachmentUri =
    "mlflow-attachment://550e8400-e29b-41d4-a716-446655440000" +
    `?content_type=text%2Fplain&trace_id=${traceId}&size=35`;
  return {
    resourceSpans: [
      {
        resource: {
          attributes: [{ key: "service.name", value: { stringValue: "t22.5-ui-smoke" } }],
        },
        scopeSpans: [
          {
            scope: { name: "t22.5-ui-smoke" },
            spans: [
              {
                traceId: traceHex,
                spanId: rootSpanHex,
                name: `answer-question-${index}`,
                startTimeUnixNano: start.toString(),
                endTimeUnixNano: (start + BigInt("500000000")).toString(),
                status: { code: "STATUS_CODE_OK" },
                attributes: [
                  { key: "mlflow.spanType", value: { stringValue: "CHAIN" } },
                  {
                    key: "mlflow.spanInputs",
                    value: {
                      stringValue: JSON.stringify({
                        question: `deterministic question ${index}`,
                        ...(index === 1 ? { attachment: attachmentUri } : {}),
                      }),
                    },
                  },
                  {
                    key: "mlflow.spanOutputs",
                    value: { stringValue: JSON.stringify({ answer: `deterministic answer ${index}` }) },
                  },
                ],
              },
              {
                traceId: traceHex,
                spanId: childSpanHex,
                parentSpanId: rootSpanHex,
                name: `deterministic-tool-${index}`,
                startTimeUnixNano: (start + BigInt("100000000")).toString(),
                endTimeUnixNano: (start + BigInt("300000000")).toString(),
                status: { code: "STATUS_CODE_OK" },
                attributes: [{ key: "mlflow.spanType", value: { stringValue: "TOOL" } }],
              },
            ],
          },
        ],
      },
    ],
  };
}

await waitForHealth();

const experiment = await request("POST", "/api/2.0/mlflow/experiments/create", {
  name: "T22.5 GenAI UI Smoke",
});
const experimentId = experiment.experiment_id;
const secondWorkspaceName = "t11-part1-secondary";
await request(
  "POST",
  "/api/3.0/mlflow/workspaces",
  { name: secondWorkspaceName, description: "T11.6 deterministic selector workspace" },
  201,
);
const traceHexes = [
  "11111111111111111111111111111111",
  "22222222222222222222222222222222",
  "33333333333333333333333333333333",
];
const traceIds = traceHexes.map((hex) => `tr-${hex}`);

for (const [index, traceHex] of traceHexes.entries()) {
  await request(
    "POST",
    "/v1/traces",
    otlpTrace(traceHex, `000000000000000${index + 1}`, `100000000000000${index + 1}`, index + 1),
    200,
    {
      "x-mlflow-experiment-id": experimentId,
    },
  );
  await request("POST", "/api/3.0/mlflow/traces", {
    trace: {
      trace_info: {
        trace_id: traceIds[index],
        trace_location: {
          type: "MLFLOW_EXPERIMENT",
          mlflow_experiment: { experiment_id: experimentId },
        },
        request_time: `2026-07-20T12:00:0${index}Z`,
        execution_duration_ms: "500",
        state: "OK",
        request_preview: JSON.stringify({ question: `deterministic question ${index + 1}` }),
        response_preview: JSON.stringify({ answer: `deterministic answer ${index + 1}` }),
        assessments:
          index === 0
            ? [
                {
                  assessment_id: "a-t22-5-correctness",
                  assessment_name: "correctness",
                  source: { source_type: "CODE", source_id: "t22.5-seed" },
                  create_time: "2026-07-20T12:00:10Z",
                  last_update_time: "2026-07-20T12:00:10Z",
                  feedback: { value: true },
                  rationale: "Deterministic seeded assessment",
                },
              ]
            : [],
        tags:
          index === 0
            ? [
                {
                  key: "mlflow.artifactLocation",
                  value: `/mlartifacts/workspaces/default/t11-traces/${traceIds[index]}`,
                },
              ]
            : [],
      },
    },
  });
}

await uploadArtifact(
  `workspaces/default/${experimentId}/traces/${traceIds[0]}/artifacts/attachments/550e8400-e29b-41d4-a716-446655440000`,
  "T11.6 deterministic trace attachment",
);

let finishedRunSequence = 0;
async function createFinishedRun(name, tags) {
  const startTime = 1784550000000 + finishedRunSequence * 1000;
  finishedRunSequence += 1;
  const created = await request("POST", "/api/2.0/mlflow/runs/create", {
    experiment_id: experimentId,
    start_time: startTime,
    tags: [{ key: "mlflow.runName", value: name }, ...tags],
  });
  const runId = created.run.info.run_id;
  await request("POST", "/api/2.0/mlflow/runs/update", {
    run_id: runId,
    status: "FINISHED",
    end_time: startTime + 1000,
  });
  return runId;
}

const evaluationRunId = await createFinishedRun("T22.5 deterministic evaluation", [
  { key: "mlflow.runType", value: "genai_evaluate" },
]);
const issueRunId = await createFinishedRun("T22.5 issue detection", [
  { key: "mlflow.runType", value: "issue_detection" },
  { key: "categories", value: "quality,correctness" },
  { key: "model", value: "fixture:/no-provider-call" },
  { key: "total_traces", value: "3" },
  { key: "mlflow.issueDetection.result.issues", value: "1" },
  { key: "mlflow.issueDetection.result.totalTracesAnalyzed", value: "3" },
  { key: "mlflow.issueDetection.result.summary", value: "Deterministic UI smoke issue summary" },
]);

const ordinaryRuns = [];
for (let index = 0; index < 110; index += 1) {
  const runStartTime =
    index < 2 ? 1784549300000 + index * 1000 : 1784549000000 + index * 1000;
  const created = await request("POST", "/api/2.0/mlflow/runs/create", {
    experiment_id: experimentId,
    start_time: runStartTime,
    run_name: index < 2 ? `T11.6 metric run ${index + 1}` : `T11.6 pagination run ${index + 1}`,
    tags: [{ key: "purpose", value: "part1-ui-smoke" }],
  });
  const runId = created.run.info.run_id;
  ordinaryRuns.push({ runId, artifactUri: created.run.info.artifact_uri });
  if (index < 2) {
    await request("POST", "/api/2.0/mlflow/runs/log-batch", {
      run_id: runId,
      metrics: Array.from({ length: 6 }, (_, step) => ({
        key: "accuracy",
        value: 0.5 + index * 0.1 + step * 0.05,
        timestamp: runStartTime + step * 100,
        step,
      })),
      params: [{ key: "optimizer", value: index === 0 ? "adam" : "sgd" }],
      tags: [{ key: "metric-fixture", value: "bulk-interval" }],
    });
  }
  await request("POST", "/api/2.0/mlflow/runs/update", {
    run_id: runId,
    status: "FINISHED",
    end_time: runStartTime + 100000,
  });
}
const metricRunIds = ordinaryRuns.slice(0, 2).map(({ runId }) => runId);
const primaryRunId = metricRunIds[0];
const primaryRunArtifactUri = ordinaryRuns[0].artifactUri;
await uploadArtifact(
  artifactProxyPath(primaryRunArtifactUri, "model/MLmodel"),
  "artifact_path: model\nflavors:\n  python_function: {}\n",
);
await uploadArtifact(
  artifactProxyPath(primaryRunArtifactUri, "notes/t11-checklist.txt"),
  "T11.6 deterministic run artifact\n",
);
await uploadArtifact(
  artifactProxyPath(ordinaryRuns[1].artifactUri, "model/MLmodel"),
  "artifact_path: model\nflavors:\n  python_function: {}\n",
);

const loggedModelResponse = await request("POST", "/api/2.0/mlflow/logged-models", {
  experiment_id: experimentId,
  name: "T11-6 deterministic logged model",
  model_type: "Classifier",
  source_run_id: primaryRunId,
  params: [{ key: "framework", value: "fixture" }],
  tags: [{ key: "purpose", value: "part1-ui-smoke" }],
});
const loggedModelId = loggedModelResponse.model.info.model_id;
await uploadArtifact(
  artifactProxyPath(loggedModelResponse.model.info.artifact_uri, "MLmodel"),
  "flavors:\n  fixture: {}\n",
);
await request("PATCH", `/api/2.0/mlflow/logged-models/${loggedModelId}`, {
  model_id: loggedModelId,
  status: "LOGGED_MODEL_READY",
});

const registeredModelName = "t11-6-registered-model";
await request("POST", "/api/2.0/mlflow/registered-models/create", {
  name: registeredModelName,
  description: "T11.6 deterministic non-prompt model",
  tags: [{ key: "purpose", value: "part1-ui-smoke" }],
});
for (let version = 1; version <= 2; version += 1) {
  await request("POST", "/api/2.0/mlflow/model-versions/create", {
    name: registeredModelName,
    source: `${primaryRunArtifactUri}/model`,
    run_id: primaryRunId,
    description: `T11.6 deterministic model version ${version}`,
    tags: [{ key: "version-fixture", value: String(version) }],
  });
}
await request("POST", "/api/2.0/mlflow/model-versions/transition-stage", {
  name: registeredModelName,
  version: "1",
  stage: "Staging",
  archive_existing_versions: false,
});
await request("POST", "/api/2.0/mlflow/registered-models/alias", {
  name: registeredModelName,
  alias: "champion",
  version: "2",
});

const issue = await request("POST", "/api/3.0/mlflow/issues", {
  experiment_id: experimentId,
  name: "T22.5 deterministic quality issue",
  description: "A seeded run-linked issue for the browser-rendered checklist.",
  status: "pending",
  severity: "medium",
  root_causes: ["prompting"],
  categories: ["quality", "correctness"],
  source_run_id: issueRunId,
  created_by: "t22.5-seed",
});

const datasetResponse = await request("POST", "/api/3.0/mlflow/datasets/create", {
  name: "T22.5 evaluation dataset",
  experiment_ids: [experimentId],
  source_type: "HUMAN",
  source: JSON.stringify({ fixture: "t22.5" }),
  schema: JSON.stringify({ type: "object" }),
  profile: JSON.stringify({ rows: 2 }),
  created_by: "t22.5-seed",
  tags: JSON.stringify({ purpose: "ui-smoke" }),
});
const datasetId = datasetResponse.dataset.dataset_id;
await request("POST", `/api/3.0/mlflow/datasets/${datasetId}/records`, {
  records: JSON.stringify([
    {
      inputs: { question: "What stack answered this request?" },
      expectations: { answer: "Rust" },
      tags: { split: "smoke" },
      source: { source_type: "HUMAN", source_data: { user: "t22.5-seed" } },
    },
    {
      inputs: { question: "Which trace is linked?" },
      expectations: { answer: traceIds[0] },
      source: { source_type: "TRACE", source_data: { trace_id: traceIds[0] } },
    },
  ]),
  updated_by: "t22.5-seed",
});

const serializedResponseLengthScorer = JSON.stringify({
  name: "T22.5 response length",
  builtin_scorer_class: "ResponseLength",
  builtin_scorer_pydantic_data: { max_length: 500, unit: "chars" },
});
const scorer = await request("POST", "/api/3.0/mlflow/scorers/register", {
  experiment_id: experimentId,
  name: "T22.5 response length",
  serialized_scorer: serializedResponseLengthScorer,
});
const scorerJob = await request("POST", "/ajax-api/3.0/mlflow/scorer/invoke", {
  experiment_id: experimentId,
  serialized_scorer: serializedResponseLengthScorer,
  trace_ids: [traceIds[0]],
  log_assessments: false,
});
const nativeScorerJobId = scorerJob.jobs?.[0]?.job_id;
if (!nativeScorerJobId) throw new Error("native scorer submission omitted jobs[0].job_id");
let nativeScorerJobStatus;
for (let attempt = 0; attempt < 60; attempt += 1) {
  const job = await request("GET", `/ajax-api/3.0/jobs/${nativeScorerJobId}`);
  nativeScorerJobStatus = job.status;
  if (nativeScorerJobStatus === "SUCCEEDED") break;
  if (["FAILED", "TIMEOUT", "CANCELED"].includes(nativeScorerJobStatus)) {
    throw new Error(`native scorer job ${nativeScorerJobId} ended as ${nativeScorerJobStatus}`);
  }
  await new Promise((resolve) => setTimeout(resolve, 250));
}
if (nativeScorerJobStatus !== "SUCCEEDED") {
  throw new Error(`native scorer job ${nativeScorerJobId} did not succeed in time`);
}

const labelSchemaResponse = await request("POST", "/api/3.0/mlflow/label-schemas/create", {
  experiment_id: experimentId,
  name: "T22.5 answer correctness",
  type: "FEEDBACK",
  input: { pass_fail: { positive_label: "Correct", negative_label: "Incorrect" } },
  instruction: "Is the deterministic answer correct?",
  enable_comment: true,
});
const labelSchemaId = labelSchemaResponse.label_schema.schema_id;
const reviewQueueResponse = await request("POST", "/api/3.0/mlflow/review-queues/create", {
  experiment_id: experimentId,
  name: "T22.5 deterministic review queue",
  queue_type: "CUSTOM",
  users: ["default"],
  schema_ids: [labelSchemaId],
});
const reviewQueueId = reviewQueueResponse.review_queue.queue_id;
await request("POST", "/api/3.0/mlflow/review-queues/items/add", {
  queue_id: reviewQueueId,
  item_type: "TRACE",
  item_ids: [traceIds[0], traceIds[1]],
});

const promptName = "t22-5-support-prompt";
await request("POST", "/api/2.0/mlflow/registered-models/create", {
  name: promptName,
  tags: [
    { key: "mlflow.prompt.is_prompt", value: "true" },
    { key: "_mlflow_experiment_ids", value: `,${experimentId},` },
    { key: "purpose", value: "ui-smoke" },
  ],
});
for (const [version, text] of [
  [1, "Answer {{question}} clearly."],
  [2, "Answer {{question}} clearly and concisely."],
]) {
  await request("POST", "/api/2.0/mlflow/model-versions/create", {
    name: promptName,
    source: "dummy-source",
    description: `T22.5 prompt version ${version}`,
    tags: [
      { key: "mlflow.prompt.is_prompt", value: "true" },
      { key: "mlflow.prompt.text", value: text },
      { key: "_mlflow_prompt_type", value: "text" },
    ],
  });
}

const secretResponse = await request("POST", "/api/3.0/mlflow/gateway/secrets/create", {
  secret_name: "t22-5-obvious-fake-secret",
  secret_value: { api_key: "obvious-fake-t22-5-value-never-used" },
  provider: "openai",
  auth_config: { auth_mode: "api_key" },
  created_by: "t22.5-seed",
});
const secretId = secretResponse.secret.secret_id;
const modelResponse = await request("POST", "/api/3.0/mlflow/gateway/model-definitions/create", {
  name: "t22-5-fake-model-definition",
  secret_id: secretId,
  provider: "openai",
  model_name: "t22-5-fake-model-never-called",
  created_by: "t22.5-seed",
});
const modelDefinitionId = modelResponse.model_definition.model_definition_id;
const endpointResponse = await request("POST", "/api/3.0/mlflow/gateway/endpoints/create", {
  name: "t22-5-deterministic-endpoint",
  model_configs: [{ model_definition_id: modelDefinitionId, linkage_type: "PRIMARY", weight: 1 }],
  routing_strategy: "REQUEST_BASED_TRAFFIC_SPLIT",
  usage_tracking: true,
  created_by: "t22.5-seed",
});
const endpointId = endpointResponse.endpoint.endpoint_id;
const guardrailScorer = await request("POST", "/api/3.0/mlflow/scorers/register", {
  experiment_id: experimentId,
  name: "T22.5 input guardrail scorer",
  serialized_scorer: JSON.stringify({
    name: "T22.5 input guardrail scorer",
    instructions_judge_pydantic_data: {
      instructions: "Reject requests that are not deterministic UI smoke inputs: {{ inputs }}",
      model: "openai:/fixture-never-called",
    },
  }),
});
const guardrailResponse = await request("POST", "/api/3.0/mlflow/gateway/guardrails/create", {
  name: "T22.5 deterministic input guardrail",
  scorer_id: guardrailScorer.scorer_id,
  scorer_version: guardrailScorer.version,
  stage: "BEFORE",
  action: "VALIDATION",
});
const guardrailId = guardrailResponse.guardrail.guardrail_id;
await request("POST", "/api/3.0/mlflow/gateway/guardrails/add-to-endpoint", {
  endpoint_id: endpointId,
  guardrail_id: guardrailId,
  execution_order: 1,
});
const budgetResponse = await request("POST", "/api/3.0/mlflow/gateway/budgets/create", {
  budget_unit: "USD",
  budget_amount: 25.5,
  duration: { unit: "DAYS", value: 1 },
  target_scope: "WORKSPACE",
  budget_action: "ALERT",
  created_by: "t22.5-seed",
});

const state = {
  baseURL,
  experimentId,
  traceIds,
  evaluationRunId,
  issueRunId,
  issueId: issue.issue.issue_id,
  datasetId,
  scorerId: scorer.scorer_id,
  nativeScorerJobId,
  labelSchemaId,
  reviewQueueId,
  promptName,
  endpointId,
  guardrailId,
  budgetPolicyId: budgetResponse.budget_policy.budget_policy_id,
  secondWorkspaceName,
  metricRunIds,
  primaryRunId,
  primaryRunArtifactUri,
  loggedModelId,
  registeredModelName,
};
await fs.writeFile(path.join(HERE, ".state.json"), `${JSON.stringify(state, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(state, null, 2)}\n`);
