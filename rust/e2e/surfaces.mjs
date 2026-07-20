export const SCREENSHOT_DIR = new URL("./screenshots/", import.meta.url);

export function surfaces(state) {
  const experiment = `/experiments/${state.experimentId}`;
  return [
    {
      id: "gateway-secrets-endpoints",
      surface: "Gateway secrets config + endpoints",
      route: "/gateway",
      assertion:
        "Gateway secrets/config request completed and the populated endpoints surface rendered; zero Python-attributed responses.",
      screenshot: "gateway-secrets-endpoints.png",
      notes:
        "The OSS page consumes secrets/config as a capability gate; when available, its rendered state is the endpoints panel.",
    },
    {
      id: "gateway-endpoint-create",
      surface: "Gateway endpoint creation",
      route: "/gateway/endpoints/create",
      assertion:
        "Create-endpoint form rendered provider/model/name controls; zero Python-attributed responses.",
      screenshot: "gateway-endpoint-create.png",
      notes: "No provider call or credential validation was submitted.",
    },
    {
      id: "gateway-budgets",
      surface: "Gateway budgets",
      route: "/gateway/budgets",
      assertion: "Budgets heading and seeded policy rendered; zero Python-attributed responses.",
      screenshot: "gateway-budgets.png",
      notes: "Populated policy state.",
    },
    {
      id: "gateway-usage",
      surface: "Gateway usage",
      route: "/gateway/usage",
      assertion: "Usage page controls rendered; zero Python-attributed responses.",
      screenshot: "gateway-usage.png",
      notes:
        "Honest empty-usage state; Rust time-bucket and percentile metric requests completed, and no live inference traffic was generated.",
    },
    {
      id: "gateway-guardrails",
      surface: "Gateway endpoint guardrails",
      route: `/gateway/endpoints/${state.endpointId}?tab=guardrails`,
      assertion:
        "Endpoint detail Guardrails tab and seeded guardrail rendered; zero Python-attributed responses.",
      screenshot: "gateway-guardrails.png",
      notes: "Guardrail was attached through Rust RPCs; no model invocation occurred.",
    },
    {
      id: "evaluation-runs",
      surface: "Evaluation runs",
      route: `${experiment}/evaluation-runs`,
      assertion:
        "Evaluation-runs page and populated deterministic run rendered; zero Python-attributed responses.",
      screenshot: "evaluation-runs.png",
      notes: "Populated state includes a completed GenAI evaluation run and issue-detection run.",
    },
    {
      id: "issues",
      surface: "Issues",
      route: `${experiment}/evaluation-runs/${state.issueRunId}/issues`,
      assertion:
        "Issue-detection run Issues tab and seeded run-linked issue rendered; zero Python-attributed responses.",
      screenshot: "issues.png",
      notes: "Populated pending issue state.",
    },
    {
      id: "datasets",
      surface: "Datasets",
      route: `${experiment}/datasets`,
      assertion: "Datasets list and seeded dataset rendered; zero Python-attributed responses.",
      screenshot: "datasets.png",
      notes: "Populated dataset state.",
    },
    {
      id: "dataset-records",
      surface: "Dataset records",
      route: `${experiment}/datasets/${state.datasetId}`,
      assertion:
        "Dataset detail and populated records table rendered; zero Python-attributed responses.",
      screenshot: "dataset-records.png",
      notes: "Two deterministic records, one linked to a seeded trace.",
    },
    {
      id: "scorers",
      surface: "Scorers / judges",
      route: `${experiment}/judges`,
      assertion:
        "Judges page and registered deterministic scorer rendered; zero Python-attributed responses.",
      screenshot: "scorers.png",
      notes:
        "Registered ResponseLength scorer and seeded a successful native worker job; no external model.",
    },
    {
      id: "review-queues",
      surface: "Review queues",
      route: `${experiment}/review-queue?selectedQueueId=${state.reviewQueueId}`,
      assertion:
        "Selected populated review queue and pending trace rendered; zero Python-attributed responses.",
      screenshot: "review-queues.png",
      notes: "Populated custom queue state.",
    },
    {
      id: "labeling",
      surface: "Labeling / focused review",
      route: `${experiment}/review-queue?selectedQueueId=${state.reviewQueueId}&selectedItemId=${state.traceIds[0]}`,
      assertion:
        "FocusedReview rendered the trace and label-schema question controls; zero Python-attributed responses.",
      screenshot: "labeling.png",
      notes: "Read-only browser smoke; it does not mutate the seeded answer.",
    },
    {
      id: "experiment-prompts",
      surface: "Experiment prompts",
      route: `${experiment}/prompts`,
      assertion:
        "Experiment-scoped prompts list and seeded prompt rendered; zero Python-attributed responses.",
      screenshot: "experiment-prompts.png",
      notes: "Prompt is linked to the seeded experiment.",
    },
    {
      id: "global-prompts",
      surface: "Global prompts",
      route: "/prompts",
      assertion:
        "Global prompts list and seeded prompt rendered; zero Python-attributed responses.",
      screenshot: "global-prompts.png",
      notes: "Populated global registry state.",
    },
    {
      id: "prompt-optimization",
      surface: "Prompt optimization",
      route: `${experiment}/prompts/${encodeURIComponent(state.promptName)}?promptVersion=2`,
      assertion:
        "Prompt details rendered and Optimize Prompt modal opened; zero Python-attributed responses.",
      screenshot: "prompt-optimization.png",
      notes: "Instruction modal only; no optimizer/provider job was submitted.",
    },
    {
      id: "assistant",
      surface: "Assistant panel (compose unauthenticated state)",
      route: `${experiment}/evaluation-runs`,
      assertion:
        "Global Assistant drawer opened and setup/unauthenticated state rendered; zero Python-attributed responses, including assistant config.",
      screenshot: "assistant.png",
      notes:
        "The expected config response was Rust-attributed 403. Authenticated CLI chat frame parity is covered by rust/compliance/recorders/test_assistant_cli_provider_differential.py (T20.2).",
    },
  ];
}
