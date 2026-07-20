export function part1Surfaces(state) {
  const experiment = `/experiments/${state.experimentId}`;
  const modelName = encodeURIComponent(state.registeredModelName);
  const metricRoute =
    `/metric/?runs=${JSON.stringify(state.metricRunIds)}` +
    `&metric=${encodeURIComponent(JSON.stringify("accuracy"))}` +
    `&experiments=${JSON.stringify([state.experimentId])}` +
    `&plot_metric_keys=${encodeURIComponent(JSON.stringify(["accuracy"]))}` +
    `&plot_layout={}&x_axis=step&y_axis_scale=linear&line_smoothness=1` +
    `&show_point=false&deselected_curves=[]&last_linear_y_axis_range=[]`;
  const rows = [
    {
      id: "experiment-list",
      surface: "Experiment list",
      route: "/experiments",
      assertion: "The seeded experiment rendered in the experiment list.",
      screenshot: "part1-experiment-list.png",
      notes: "Populated tracking state.",
    },
    {
      id: "runs-table-pagination",
      surface: "Runs table + Load more",
      route: `${experiment}/runs`,
      assertion: "The runs grid rendered more than one page and Load more fetched the next client page.",
      screenshot: "part1-runs-load-more.png",
      notes: "The seed creates 110 ordinary runs; the UI requests 100 per page.",
    },
    {
      id: "run-detail-graphql",
      surface: "Run detail (GraphQL)",
      route: `${experiment}/runs/${state.primaryRunId}`,
      assertion: "Run overview rendered the seeded run and its GraphQL request completed from Rust.",
      screenshot: "part1-run-detail.png",
      notes: "GraphQL run-details feature flag is enabled in the production OSS build.",
    },
    {
      id: "charts-bulk-interval",
      surface: "Run charts (bulk interval)",
      route: `${experiment}/runs?compareRunsMode=CHART`,
      assertion: "The accuracy chart rendered and Rust served interval-sampled histories for both metric runs.",
      screenshot: "part1-runs-charts.png",
      notes: "Each metric run has six deterministic steps.",
    },
    {
      id: "compare-runs",
      surface: "Compare runs",
      route: `/compare-runs?runs=${JSON.stringify(state.metricRunIds)}&experiments=${JSON.stringify([
        state.experimentId,
      ])}`,
      assertion: "The two seeded runs rendered on the comparison page with their accuracy data.",
      screenshot: "part1-compare-runs.png",
      notes: "Direct route uses the production getCompareRunPageRoute shape.",
    },
    {
      id: "metric-page",
      surface: "Metric page",
      route: metricRoute,
      assertion: "The accuracy metric plot rendered histories for both seeded runs.",
      screenshot: "part1-metric-page.png",
      notes: "Direct route uses the production getMetricPageRoute query shape.",
    },
    {
      id: "artifact-browser",
      surface: "Run artifact browser",
      route: `${experiment}/runs/${state.primaryRunId}/artifacts`,
      assertion: "The uploaded model directory and deterministic artifact rendered in the run artifact browser.",
      screenshot: "part1-run-artifacts.png",
      notes: "Artifact bytes were uploaded through the Rust proxy plane.",
    },
    {
      id: "traces-list",
      surface: "Traces tab — list",
      route: `${experiment}/traces`,
      assertion: "The populated traces table rendered deterministic request previews.",
      screenshot: "part1-traces-list.png",
      notes: "The same real OTLP traces support the detail checks.",
    },
    {
      id: "trace-span-tree",
      surface: "Trace detail — span tree",
      route: `${experiment}/traces/${state.traceIds[0]}`,
      assertion: "The trace detail rendered the root and child spans.",
      screenshot: "part1-trace-span-tree.png",
      notes: "Two-level OTLP span tree.",
    },
    {
      id: "trace-attachment",
      surface: "Trace detail — attachment",
      route: `${experiment}/traces/${state.traceIds[0]}`,
      assertion: "The attachment URI rendered and Rust returned the seeded attachment bytes.",
      screenshot: "part1-trace-attachment.png",
      notes: "Real text attachment stored below the trace artifact root.",
    },
    {
      id: "trace-assessment",
      surface: "Trace detail — assessment",
      route: `${experiment}/traces/${state.traceIds[0]}`,
      assertion: "The seeded correctness assessment name and true feedback value rendered.",
      screenshot: "part1-trace-assessment.png",
      notes: "Read-only smoke of the real assessment record.",
    },
    {
      id: "logged-models-list",
      surface: "Logged models tab",
      route: `${experiment}/models`,
      assertion: "The finalized logged model rendered in the experiment model list.",
      screenshot: "part1-logged-models.png",
      notes: "Created and finalized through /api/2.0/mlflow/logged-models.",
    },
    {
      id: "logged-model-detail",
      surface: "Logged model detail",
      route: `${experiment}/models/${state.loggedModelId}`,
      assertion: "The finalized model detail rendered its ID, name, status, and parameters.",
      screenshot: "part1-logged-model-detail.png",
      notes: "Populated finalized state.",
    },
    {
      id: "dataset-records",
      surface: "Datasets dropdown + records",
      route: `${experiment}/datasets/${state.datasetId}`,
      assertion: "The experiment Datasets navigation and populated dataset records rendered.",
      screenshot: "dataset-records.png",
      notes: "Shared with the T22.5 test; intentionally not duplicated.",
    },
    {
      id: "registry-models-list",
      surface: "Model Registry — models list",
      route: "/models",
      assertion: "The seeded non-prompt registered model rendered in the global registry list.",
      screenshot: "part1-registry-models.png",
      notes: "Prompt-tagged models remain excluded from this assertion.",
    },
    {
      id: "registry-model-overview",
      surface: "Model Registry — model, versions, stages",
      route: `/models/${modelName}`,
      assertion: "The model overview rendered two versions, then version 1 rendered the seeded Staging transition.",
      screenshot: "part1-registry-model.png",
      notes: "Both version rows are backed by real run artifacts.",
    },
    {
      id: "registry-version-detail",
      surface: "Model Registry — version, alias, artifact download",
      route: `/models/${modelName}/versions/2`,
      assertion: "Version 2 rendered the champion alias; its Source Run link opened the artifact UI and downloaded MLmodel from Rust.",
      screenshot: "part1-registry-version.png",
      notes: "The test verifies the response body, not only the download control.",
    },
    {
      id: "experiment-prompts",
      surface: "Experiment prompts",
      route: `${experiment}/prompts`,
      assertion: "The experiment-scoped prompt list rendered the seeded prompt.",
      screenshot: "experiment-prompts.png",
      notes: "Shared with the T22.5 test; intentionally not duplicated.",
    },
    {
      id: "global-prompts",
      surface: "Global prompts",
      route: "/prompts",
      assertion: "The global prompt registry rendered the seeded prompt.",
      screenshot: "global-prompts.png",
      notes: "Shared with the T22.5 test; intentionally not duplicated.",
    },
    {
      id: "prompt-optimization",
      surface: "Prompt detail + optimization entry",
      route: `${experiment}/prompts/${encodeURIComponent(state.promptName)}?promptVersion=2`,
      assertion: "Prompt version detail rendered and the deterministic optimization instruction modal opened.",
      screenshot: "prompt-optimization.png",
      notes: "Shared with T22.5; no provider job was submitted.",
    },
    {
      id: "workspace-selector",
      surface: "Workspace selector",
      route: "/experiments",
      assertion: "The production sidebar selector listed default and the seeded second workspace and switched context.",
      screenshot: "part1-workspace-selector.png",
      notes: "Fully enabled by Rust server-info from --enable-workspaces.",
    },
  ];

  return rows.map((row) => ({ ...row, section: "T11.6 auth-disabled Part 1" }));
}

export function authSurfaces() {
  return [
    {
      id: "admin-users-crud",
      surface: "Admin console — user CRUD",
      route: "/admin",
      assertion: "Created a disposable user through the UI, rendered its detail, then deleted it through the UI.",
      screenshot: "auth-admin-user.png",
      notes: "Bootstrap admin/password1234; cleanup verified in the users table.",
    },
    {
      id: "admin-roles-crud",
      surface: "Admin console — role CRUD",
      route: "/admin?tab=roles",
      assertion: "Created a disposable role through the UI, rendered its detail, then deleted it through the UI.",
      screenshot: "auth-admin-role.png",
      notes: "Role belongs to the default workspace.",
    },
    {
      id: "admin-edit-access",
      surface: "Admin console — EditAccessModal grant",
      route: "/admin/users/t11-ui-user",
      assertion: "Assigned the disposable role to the user through EditAccessModal and verified the role on user detail.",
      screenshot: "auth-edit-access.png",
      notes: "Functional role grant, not a render-only modal check.",
    },
    {
      id: "account-permissions",
      surface: "Account — current-user permissions",
      route: "/account?tab=permissions",
      assertion: "The admin identity and its current-user permissions table rendered from authenticated Rust APIs.",
      screenshot: "auth-account-permissions.png",
      notes: "/users/current is asserted 200 with is_basic_auth=true.",
    },
    {
      id: "basic-auth-logout",
      surface: "Basic-auth logout behavior",
      route: "/account",
      assertion: "Logout appeared only after is_basic_auth=true; clicking it cleared auth cookies and issued the deliberate bogus-credential users/current XHR.",
      screenshot: "auth-logout.png",
      notes: "The one deliberate logout 401 is dynamically allowlisted; all normal authenticated probes must be 200.",
    },
  ].map((row) => ({ ...row, section: "T9.9 auth-enabled admin/account" }));
}
