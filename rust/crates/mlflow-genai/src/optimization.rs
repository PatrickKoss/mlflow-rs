//! Native prompt optimization compatible with MLflow's pinned MetaPrompt and
//! GEPA 0.0.27 execution paths.
//!
//! The state machine is deliberately independent from the worker's HTTP store
//! adapter. Tests inject a scripted [`OptimizationRuntime`], while production
//! jobs use the same engine with the native scorer executor and tracking APIs.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use async_trait::async_trait;
use chrono::Utc;
use reqwest::Method;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    supported_builtin_scorers, EngineError, EvalItem, ScorerExecutor, SerializedScorer,
    WorkerRequest,
};

const PROMPT_TEXT_TAG: &str = "mlflow.prompt.text";
const PROMPT_TYPE_TAG: &str = "_mlflow_prompt_type";
const PROMPT_MODEL_CONFIG_TAG: &str = "_mlflow_prompt_model_config";
const IS_PROMPT_TAG: &str = "mlflow.prompt.is_prompt";
const LINKED_PROMPTS_TAG: &str = "mlflow.linkedPrompts";
const LOGGED_ARTIFACTS_TAG: &str = "mlflow.loggedArtifacts";
const PINNED_MLFLOW_VERSION: &str = "3.14.1.dev0";

const GEPA_REFLECTION_TEMPLATE: &str = r#"I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."#;

const META_PROMPT_TEMPLATE: &str = r#"You are an expert prompt engineer. Your task is to improve
the following prompts to achieve better performance.

CURRENT PROMPTS:
{current_prompts_formatted}

{evaluation_examples}

PROMPT ENGINEERING BEST PRACTICES:
Apply these proven techniques to create effective prompts:

1. **Clarity & Specificity**: Be explicit about the task, expected output format,
and any constraints
2. **Structured Formatting**: Use numbered lists, sections, or delimiters to
organize complex instructions clearly
3. **Few-Shot Examples**: Include concrete examples showing desired input/output
pairs when appropriate
4. **Role/Persona**: Specify expertise level if relevant (e.g., "You are an expert
mathematician...")
5. **Step-by-Step Decomposition**: Break complex reasoning tasks into explicit
steps or phases
6. **Output Format Specification**: Explicitly define the format, structure, and
constraints for outputs
7. **Constraint Specification**: Clearly state what to avoid, exclude, or not do
8. **Verification Instructions**: Add self-checking steps for calculation-heavy
or error-prone tasks
9. **Chain-of-Thought Prompting**: For reasoning tasks, explicitly instruct to
show intermediate steps

CRITICAL REQUIREMENT - TEMPLATE VARIABLES:
The following variables MUST be preserved EXACTLY as shown in the original prompts.
DO NOT modify, remove, add, or change the formatting of these variables in any way:
{template_variables}

IMPORTANT: Template variables use double curly braces like {{{{variable_name}}}}.
You MUST copy them exactly as they appear in the original prompt into your improved
prompt. If a variable appears as {{{{question}}}} in the original, it must appear as
{{{{question}}}} in your improvement.
If a prompt has NO template variables (marked as 'none' above), you MUST NOT introduce
any new {{{{...}}}} patterns. Only improve the wording and phrasing of those prompts.

{custom_guidelines}

INSTRUCTIONS:
Generate improved versions of the prompts by applying relevant prompt engineering
best practices. Make your prompts specific and actionable.

{extra_instructions}

CRITICAL: Preserve all template variables in their exact original format with
double curly braces.

CRITICAL: You must respond with a valid JSON object using the EXACT prompt names
shown above. The JSON keys must match the "Prompt name" fields exactly. Use this
structure:
{{
{response_format_example}
}}

REMINDER:
1. Use the exact prompt names as JSON keys (e.g., if the prompt is named
"aime_solver", use "aime_solver" as the key)
2. Every template variable from the original prompt must appear unchanged in your
improved version
3. Apply best practices that are most relevant to the task at hand

Do not include any text before or after the JSON object. Do not include
explanations or reasoning.
"#;

/// Insertion-ordered prompt component map, matching Python dictionaries.
pub type Candidate = Map<String, Value>;

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationRecord {
    pub inputs: Value,
    pub outputs: Value,
    pub expectations: Value,
    pub score: Option<f64>,
    pub rationales: Map<String, Value>,
    pub individual_scores: Map<String, Value>,
    pub trace_spans: Vec<Value>,
}

#[async_trait]
pub trait OptimizationRuntime: Send {
    async fn evaluate(
        &mut self,
        candidate: &Candidate,
        data: &[Value],
        capture_traces: bool,
    ) -> Result<Vec<EvaluationRecord>, EngineError>;

    async fn reflect(
        &mut self,
        model: &str,
        prompt: &str,
        json_mode: bool,
        inference: Option<&Map<String, Value>>,
    ) -> Result<String, EngineError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetaPromptConfig {
    pub reflection_model: String,
    pub lm_kwargs: Map<String, Value>,
    pub guidelines: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GepaConfig {
    pub reflection_model: String,
    pub max_metric_calls: i64,
    pub gepa_kwargs: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerConfig {
    MetaPrompt(MetaPromptConfig),
    Gepa(GepaConfig),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationOutput {
    pub optimized_prompts: Candidate,
    pub initial_eval_score: Option<f64>,
    pub final_eval_score: Option<f64>,
    pub initial_eval_score_per_scorer: Map<String, Value>,
    pub final_eval_score_per_scorer: Map<String, Value>,
    pub candidate_sequence: Vec<Candidate>,
    pub selected_candidate_indices: Vec<usize>,
    pub validation_scores: Vec<f64>,
    pub metric_calls: usize,
}

pub struct OptimizationEngine<R> {
    runtime: R,
}

impl<R: OptimizationRuntime> OptimizationEngine<R> {
    pub fn new(runtime: R) -> Self {
        Self { runtime }
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.runtime
    }

    pub async fn optimize(
        &mut self,
        config: &OptimizerConfig,
        train_data: &[Value],
        target_prompts: &Candidate,
    ) -> Result<OptimizationOutput, EngineError> {
        validate_candidate(target_prompts)?;
        match config {
            OptimizerConfig::MetaPrompt(config) => {
                self.optimize_metaprompt(config, train_data, target_prompts)
                    .await
            }
            OptimizerConfig::Gepa(config) => {
                self.optimize_gepa(config, train_data, target_prompts).await
            }
        }
    }

    async fn optimize_metaprompt(
        &mut self,
        config: &MetaPromptConfig,
        train_data: &[Value],
        target_prompts: &Candidate,
    ) -> Result<OptimizationOutput, EngineError> {
        let variables = template_variables(target_prompts);
        if train_data.is_empty() {
            let meta_prompt = build_meta_prompt(config, target_prompts, &variables, None)?;
            let improved = self
                .runtime
                .reflect(
                    &config.reflection_model,
                    &meta_prompt,
                    true,
                    (!config.lm_kwargs.is_empty()).then_some(&config.lm_kwargs),
                )
                .await
                .and_then(|content| parse_metaprompt_response(&content))
                .and_then(|mut candidate| {
                    validate_prompt_names(target_prompts, &candidate)?;
                    validate_template_variables(target_prompts, &mut candidate)?;
                    Ok(candidate)
                })
                .unwrap_or_else(|_| target_prompts.clone());
            return Ok(OptimizationOutput {
                optimized_prompts: improved.clone(),
                initial_eval_score: None,
                final_eval_score: None,
                initial_eval_score_per_scorer: Map::new(),
                final_eval_score_per_scorer: Map::new(),
                candidate_sequence: vec![target_prompts.clone(), improved],
                selected_candidate_indices: Vec::new(),
                validation_scores: Vec::new(),
                metric_calls: 0,
            });
        }

        let baseline = self
            .runtime
            .evaluate(target_prompts, train_data, false)
            .await?;
        let initial = aggregate_score(&baseline);
        let initial_per_scorer = per_scorer_scores(&baseline)?;
        let meta_prompt = build_meta_prompt(config, target_prompts, &variables, Some(&baseline))?;
        let improved = match self
            .runtime
            .reflect(
                &config.reflection_model,
                &meta_prompt,
                true,
                (!config.lm_kwargs.is_empty()).then_some(&config.lm_kwargs),
            )
            .await
            .and_then(|content| parse_metaprompt_response(&content))
            .and_then(|mut candidate| {
                validate_prompt_names(target_prompts, &candidate)?;
                validate_template_variables(target_prompts, &mut candidate)?;
                Ok(candidate)
            }) {
            Ok(candidate) => candidate,
            Err(_) => {
                return Ok(OptimizationOutput {
                    optimized_prompts: target_prompts.clone(),
                    initial_eval_score: initial,
                    final_eval_score: None,
                    initial_eval_score_per_scorer: initial_per_scorer,
                    final_eval_score_per_scorer: Map::new(),
                    candidate_sequence: vec![target_prompts.clone()],
                    selected_candidate_indices: Vec::new(),
                    validation_scores: initial.into_iter().collect(),
                    metric_calls: train_data.len(),
                });
            }
        };
        let (final_score, final_per_scorer, metric_calls) = if initial.is_some() {
            let final_results = self.runtime.evaluate(&improved, train_data, false).await?;
            (
                aggregate_score(&final_results),
                per_scorer_scores(&final_results)?,
                train_data.len() * 2,
            )
        } else {
            (None, Map::new(), train_data.len())
        };
        Ok(OptimizationOutput {
            optimized_prompts: improved.clone(),
            initial_eval_score: initial,
            final_eval_score: final_score,
            initial_eval_score_per_scorer: initial_per_scorer,
            final_eval_score_per_scorer: final_per_scorer,
            candidate_sequence: vec![target_prompts.clone(), improved],
            selected_candidate_indices: Vec::new(),
            validation_scores: [initial, final_score].into_iter().flatten().collect(),
            metric_calls,
        })
    }

    async fn optimize_gepa(
        &mut self,
        config: &GepaConfig,
        train_data: &[Value],
        target_prompts: &Candidate,
    ) -> Result<OptimizationOutput, EngineError> {
        if train_data.is_empty() {
            return Err(EngineError::InvalidParams(
                "GEPA optimizer requires `train_data` to be provided.".to_string(),
            ));
        }
        let options = GepaOptions::parse(&config.gepa_kwargs)?;
        let mut state = GepaState::new(target_prompts.clone(), options.frontier_type);
        let baseline = self
            .runtime
            .evaluate(target_prompts, train_data, false)
            .await?;
        let baseline_scores = required_scores(&baseline)?;
        state.add_initial(
            baseline_scores,
            objective_scores(&baseline),
            train_data.len(),
        )?;

        let mut rng = PythonRandom::new(options.seed);
        let mut sampler = EpochSampler::new(options.reflection_minibatch_size);
        while state.metric_calls < usize::try_from(config.max_metric_calls).unwrap_or_default() {
            state.iteration += 1;
            let selected = select_candidate(&state, &options, &mut rng)?;
            state.selected.push(selected);
            let minibatch_ids = sampler.next(train_data.len(), state.iteration - 1, &mut rng)?;
            let minibatch = minibatch_ids
                .iter()
                .map(|index| train_data[*index].clone())
                .collect::<Vec<_>>();
            let current = state.candidates[selected].clone();
            let current_eval = self.runtime.evaluate(&current, &minibatch, true).await?;
            state.metric_calls += minibatch_ids.len();
            let current_scores = required_scores(&current_eval)?;
            if options.skip_perfect_score
                && current_scores
                    .iter()
                    .all(|score| *score >= options.perfect_score)
            {
                continue;
            }

            let components = state.components_to_update(selected, &options.module_selector)?;
            let reflective = reflective_dataset(&current, &current_eval, &components);
            let mut proposed = current.clone();
            let mut proposal_failed = false;
            for component in components {
                let Some(records) = reflective.get(&component).and_then(Value::as_array) else {
                    continue;
                };
                if records.is_empty() {
                    continue;
                }
                let prompt = render_gepa_reflection(
                    candidate_text(&current, &component)?,
                    records,
                    options.reflection_prompt_template.as_deref(),
                );
                match self
                    .runtime
                    .reflect(&config.reflection_model, &prompt, false, None)
                    .await
                {
                    Ok(text) => {
                        proposed.insert(component, Value::String(extract_instruction(&text)));
                    }
                    Err(_) => {
                        proposal_failed = true;
                        break;
                    }
                }
            }
            if proposal_failed {
                continue;
            }
            let proposed_eval = self.runtime.evaluate(&proposed, &minibatch, false).await?;
            state.metric_calls += minibatch_ids.len();
            let proposed_scores = required_scores(&proposed_eval)?;
            if proposed_scores.iter().sum::<f64>() <= current_scores.iter().sum::<f64>() {
                continue;
            }
            let validation = self.runtime.evaluate(&proposed, train_data, false).await?;
            state.metric_calls += train_data.len();
            state.add_candidate(
                proposed,
                vec![selected],
                required_scores(&validation)?,
                objective_scores(&validation),
            )?;
        }

        let best = state.best_candidate();
        let initial_score = state.validation_scores.first().copied();
        let final_score = state.validation_scores.iter().copied().reduce(f64::max);
        let best_idx = state
            .validation_scores
            .iter()
            .position(|score| Some(*score) == final_score)
            .unwrap_or_default();
        Ok(OptimizationOutput {
            optimized_prompts: state.candidates[best].clone(),
            initial_eval_score: initial_score,
            final_eval_score: final_score,
            initial_eval_score_per_scorer: state
                .objective_averages
                .first()
                .cloned()
                .unwrap_or_default(),
            final_eval_score_per_scorer: state
                .objective_averages
                .get(best_idx)
                .cloned()
                .unwrap_or_default(),
            candidate_sequence: state.candidates,
            selected_candidate_indices: state.selected,
            validation_scores: state.validation_scores,
            metric_calls: state.metric_calls,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrontierType {
    Instance,
    Objective,
    Hybrid,
    Cartesian,
}

#[derive(Debug, Clone)]
struct GepaOptions {
    candidate_selection_strategy: String,
    frontier_type: FrontierType,
    skip_perfect_score: bool,
    reflection_minibatch_size: usize,
    perfect_score: f64,
    reflection_prompt_template: Option<String>,
    module_selector: String,
    seed: i64,
}

impl GepaOptions {
    fn parse(raw: &Map<String, Value>) -> Result<Self, EngineError> {
        const ALLOWED: &[&str] = &[
            "seed_candidate",
            "trainset",
            "valset",
            "adapter",
            "task_lm",
            "evaluator",
            "reflection_lm",
            "candidate_selection_strategy",
            "frontier_type",
            "skip_perfect_score",
            "batch_sampler",
            "reflection_minibatch_size",
            "perfect_score",
            "reflection_prompt_template",
            "module_selector",
            "use_merge",
            "max_merge_invocations",
            "merge_val_overlap_floor",
            "max_metric_calls",
            "stop_callbacks",
            "logger",
            "run_dir",
            "callbacks",
            "use_wandb",
            "wandb_api_key",
            "wandb_init_kwargs",
            "use_mlflow",
            "mlflow_tracking_uri",
            "mlflow_experiment_name",
            "track_best_outputs",
            "display_progress_bar",
            "use_cloudpickle",
            "cache_evaluation",
            "seed",
            "raise_on_exception",
            "val_evaluation_policy",
        ];
        if let Some(unknown) = raw.keys().find(|key| !ALLOWED.contains(&key.as_str())) {
            return Err(EngineError::InvalidParams(format!(
                "optimize() got an unexpected keyword argument '{unknown}'"
            )));
        }
        let strategy = string_option(raw, "candidate_selection_strategy", "pareto")?;
        if !["pareto", "current_best", "epsilon_greedy"].contains(&strategy.as_str()) {
            return Err(EngineError::InvalidParams(format!(
                "Unknown candidate_selector strategy: {strategy}. Supported strategies: 'pareto', 'current_best', 'epsilon_greedy'"
            )));
        }
        let frontier = string_option(raw, "frontier_type", "instance")?;
        let frontier_type = match frontier.as_str() {
            "instance" => FrontierType::Instance,
            "objective" => FrontierType::Objective,
            "hybrid" => FrontierType::Hybrid,
            "cartesian" => FrontierType::Cartesian,
            _ => {
                return Err(EngineError::InvalidParams(format!(
                    "Unknown frontier_type: {frontier}"
                )))
            }
        };
        let module_selector = string_option(raw, "module_selector", "round_robin")?;
        if !["round_robin", "all"].contains(&module_selector.as_str()) {
            return Err(EngineError::InvalidParams(format!(
                "Unknown module_selector strategy: {module_selector}. Supported strategies: 'round_robin', 'all'"
            )));
        }
        if raw
            .get("batch_sampler")
            .is_some_and(|value| value != "epoch_shuffled")
        {
            return Err(EngineError::InvalidParams(
                "Only the pinned 'epoch_shuffled' batch_sampler is serializable for server jobs"
                    .to_string(),
            ));
        }
        let reflection_prompt_template = optional_string(raw, "reflection_prompt_template")?;
        if let Some(template) = &reflection_prompt_template {
            let missing = ["<curr_instructions>", "<inputs_outputs_feedback>"]
                .into_iter()
                .filter(|placeholder| !template.contains(placeholder))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(EngineError::InvalidParams(format!(
                    "Missing placeholder(s) in prompt template: {}",
                    missing.join(", ")
                )));
            }
        }
        Ok(Self {
            candidate_selection_strategy: strategy,
            frontier_type,
            skip_perfect_score: bool_option(raw, "skip_perfect_score", true)?,
            reflection_minibatch_size: usize_option(raw, "reflection_minibatch_size", 3)?,
            perfect_score: number_option(raw, "perfect_score", 1.0)?,
            reflection_prompt_template,
            module_selector,
            seed: integer_option(raw, "seed", 0)?,
        })
    }
}

struct GepaState {
    candidates: Vec<Candidate>,
    parents: Vec<Vec<Option<usize>>>,
    validation_subscores: Vec<Vec<f64>>,
    objective_averages: Vec<Map<String, Value>>,
    validation_scores: Vec<f64>,
    instance_front: Vec<BTreeSet<usize>>,
    objective_front: BTreeMap<String, BTreeSet<usize>>,
    cartesian_front: BTreeMap<(usize, String), BTreeSet<usize>>,
    next_component: Vec<usize>,
    components: Vec<String>,
    frontier_type: FrontierType,
    metric_calls: usize,
    iteration: usize,
    selected: Vec<usize>,
}

impl GepaState {
    fn new(seed: Candidate, frontier_type: FrontierType) -> Self {
        let components = seed.keys().cloned().collect();
        Self {
            candidates: vec![seed],
            parents: vec![vec![None]],
            validation_subscores: Vec::new(),
            objective_averages: Vec::new(),
            validation_scores: Vec::new(),
            instance_front: Vec::new(),
            objective_front: BTreeMap::new(),
            cartesian_front: BTreeMap::new(),
            next_component: vec![0],
            components,
            frontier_type,
            metric_calls: 0,
            iteration: 0,
            selected: Vec::new(),
        }
    }

    fn add_initial(
        &mut self,
        scores: Vec<f64>,
        objectives: Vec<Map<String, Value>>,
        calls: usize,
    ) -> Result<(), EngineError> {
        self.metric_calls = calls;
        self.validation_subscores.push(scores.clone());
        self.validation_scores.push(mean(&scores));
        let averages = average_objectives(&objectives)?;
        self.objective_averages.push(averages.clone());
        self.instance_front = (0..scores.len()).map(|_| BTreeSet::from([0])).collect();
        for name in averages.keys() {
            self.objective_front
                .insert(name.clone(), BTreeSet::from([0]));
        }
        self.validate_objectives(&objectives)?;
        if self.frontier_type == FrontierType::Cartesian {
            for (index, objective) in objectives.iter().enumerate() {
                for name in objective.keys() {
                    self.cartesian_front
                        .insert((index, name.clone()), BTreeSet::from([0]));
                }
            }
        }
        Ok(())
    }

    fn add_candidate(
        &mut self,
        candidate: Candidate,
        parents: Vec<usize>,
        scores: Vec<f64>,
        objectives: Vec<Map<String, Value>>,
    ) -> Result<(), EngineError> {
        self.validate_objectives(&objectives)?;
        let index = self.candidates.len();
        let next = parents
            .iter()
            .filter_map(|parent| self.next_component.get(*parent))
            .copied()
            .max()
            .unwrap_or_default();
        self.candidates.push(candidate);
        self.parents
            .push(parents.into_iter().map(Some).collect::<Vec<_>>());
        self.next_component.push(next);
        for (example, score) in scores.iter().enumerate() {
            let best = self
                .validation_subscores
                .iter()
                .map(|values| values[example])
                .fold(f64::NEG_INFINITY, f64::max);
            if *score > best {
                self.instance_front[example] = BTreeSet::from([index]);
            } else if *score == best {
                self.instance_front[example].insert(index);
            }
        }
        let averages = average_objectives(&objectives)?;
        for (name, value) in &averages {
            let value = value.as_f64().unwrap_or(f64::NEG_INFINITY);
            let best = self
                .objective_averages
                .iter()
                .filter_map(|scores| scores.get(name).and_then(Value::as_f64))
                .fold(f64::NEG_INFINITY, f64::max);
            if value > best {
                self.objective_front
                    .insert(name.clone(), BTreeSet::from([index]));
            } else if value == best {
                self.objective_front
                    .entry(name.clone())
                    .or_default()
                    .insert(index);
            }
        }
        if self.frontier_type == FrontierType::Cartesian {
            for (example, objective) in objectives.iter().enumerate() {
                for (name, value) in objective {
                    let score = value.as_f64().unwrap_or(f64::NEG_INFINITY);
                    let best = self
                        .cartesian_front
                        .get(&(example, name.clone()))
                        .and_then(|front| front.iter().next())
                        .and_then(|candidate| {
                            self.objective_averages[*candidate]
                                .get(name)
                                .and_then(Value::as_f64)
                        })
                        .unwrap_or(f64::NEG_INFINITY);
                    if score > best {
                        self.cartesian_front
                            .insert((example, name.clone()), BTreeSet::from([index]));
                    } else if score == best {
                        self.cartesian_front
                            .entry((example, name.clone()))
                            .or_default()
                            .insert(index);
                    }
                }
            }
        }
        self.validation_scores.push(mean(&scores));
        self.validation_subscores.push(scores);
        self.objective_averages.push(averages);
        Ok(())
    }

    fn validate_objectives(&self, objectives: &[Map<String, Value>]) -> Result<(), EngineError> {
        if self.frontier_type != FrontierType::Instance && objectives.iter().all(Map::is_empty) {
            return Err(EngineError::InvalidParams(format!(
                "frontier_type='{}' requires objective_scores to be provided by the evaluator, but none were found{}.",
                match self.frontier_type {
                    FrontierType::Objective => "objective",
                    FrontierType::Hybrid => "hybrid",
                    FrontierType::Cartesian => "cartesian",
                    FrontierType::Instance => unreachable!(),
                },
                if self.validation_subscores.is_empty() {
                    ". Use an evaluator that returns objective_scores or use frontier_type='instance'"
                } else {
                    " in the evaluation result"
                }
            )));
        }
        Ok(())
    }

    fn fronts(&self) -> Vec<BTreeSet<usize>> {
        match self.frontier_type {
            FrontierType::Instance => self.instance_front.clone(),
            FrontierType::Objective => self.objective_front.values().cloned().collect(),
            FrontierType::Hybrid => self
                .instance_front
                .iter()
                .cloned()
                .chain(self.objective_front.values().cloned())
                .collect(),
            FrontierType::Cartesian => self.cartesian_front.values().cloned().collect(),
        }
    }

    fn components_to_update(
        &mut self,
        candidate: usize,
        selector: &str,
    ) -> Result<Vec<String>, EngineError> {
        if selector == "all" {
            return Ok(self.components.clone());
        }
        if self.components.is_empty() {
            return Err(EngineError::InvalidParams(
                "seed_candidate must contain at least one component text.".to_string(),
            ));
        }
        let index = self.next_component[candidate];
        self.next_component[candidate] = (index + 1) % self.components.len();
        Ok(vec![self.components[index].clone()])
    }

    fn best_candidate(&self) -> usize {
        let mut best = 0;
        for index in 1..self.validation_scores.len() {
            if self.validation_scores[index] > self.validation_scores[best] {
                best = index;
            }
        }
        best
    }
}

fn select_candidate(
    state: &GepaState,
    options: &GepaOptions,
    rng: &mut PythonRandom,
) -> Result<usize, EngineError> {
    match options.candidate_selection_strategy.as_str() {
        "current_best" => Ok(state.best_candidate()),
        "epsilon_greedy" => {
            if rng.random() < 0.1 {
                Ok(rng.randbelow(state.candidates.len()))
            } else {
                Ok(state.best_candidate())
            }
        }
        "pareto" => {
            let fronts = remove_dominated(state.fronts(), &state.validation_scores);
            let mut frequencies = Vec::<(usize, usize)>::new();
            for front in fronts {
                for candidate in front {
                    if let Some((_, count)) = frequencies
                        .iter_mut()
                        .find(|(existing, _)| *existing == candidate)
                    {
                        *count += 1;
                    } else {
                        frequencies.push((candidate, 1));
                    }
                }
            }
            let sampling = frequencies
                .into_iter()
                .flat_map(|(candidate, count)| std::iter::repeat_n(candidate, count))
                .collect::<Vec<_>>();
            if sampling.is_empty() {
                return Err(EngineError::InvalidParams(
                    "GEPA Pareto candidate sampling list is empty".to_string(),
                ));
            }
            Ok(sampling[rng.randbelow(sampling.len())])
        }
        _ => unreachable!("strategy validated while parsing"),
    }
}

fn remove_dominated(fronts: Vec<BTreeSet<usize>>, scores: &[f64]) -> Vec<BTreeSet<usize>> {
    let mut programs = fronts
        .iter()
        .flat_map(BTreeSet::iter)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    programs.sort_by(|left, right| {
        scores[*left]
            .partial_cmp(&scores[*right])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut dominated = BTreeSet::new();
    loop {
        let mut removed = false;
        for candidate in &programs {
            if dominated.contains(candidate) {
                continue;
            }
            let remaining = programs
                .iter()
                .copied()
                .filter(|other| other != candidate && !dominated.contains(other))
                .collect::<BTreeSet<_>>();
            let candidate_fronts = fronts.iter().filter(|front| front.contains(candidate));
            if candidate_fronts
                .clone()
                .all(|front| front.iter().any(|other| remaining.contains(other)))
            {
                dominated.insert(*candidate);
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }
    fronts
        .into_iter()
        .map(|front| {
            front
                .into_iter()
                .filter(|candidate| !dominated.contains(candidate))
                .collect()
        })
        .collect()
}

struct EpochSampler {
    minibatch_size: usize,
    shuffled: Vec<usize>,
    epoch: Option<usize>,
    trainset_size: usize,
}

impl EpochSampler {
    fn new(minibatch_size: usize) -> Self {
        Self {
            minibatch_size,
            shuffled: Vec::new(),
            epoch: None,
            trainset_size: 0,
        }
    }

    fn next(
        &mut self,
        trainset_size: usize,
        iteration_zero_based: usize,
        rng: &mut PythonRandom,
    ) -> Result<Vec<usize>, EngineError> {
        if trainset_size == 0 {
            return Err(EngineError::InvalidParams(
                "Cannot sample a minibatch from an empty loader.".to_string(),
            ));
        }
        if self.minibatch_size == 0 {
            return Err(EngineError::InvalidParams(
                "integer modulo by zero".to_string(),
            ));
        }
        let base = iteration_zero_based * self.minibatch_size;
        let current_epoch = self.epoch.map_or(0, |_| base / self.shuffled.len().max(1));
        if self.shuffled.is_empty()
            || trainset_size != self.trainset_size
            || self.epoch.is_some_and(|epoch| current_epoch > epoch)
        {
            self.epoch = Some(current_epoch);
            self.trainset_size = trainset_size;
            self.shuffled = (0..trainset_size).collect();
            rng.shuffle(&mut self.shuffled);
            let padding =
                (self.minibatch_size - trainset_size % self.minibatch_size) % self.minibatch_size;
            let mut frequencies = vec![1_usize; trainset_size];
            for _ in 0..padding {
                let minimum = *frequencies.iter().min().expect("dataset is non-empty");
                let selected = frequencies
                    .iter()
                    .enumerate()
                    .filter(|(_, frequency)| **frequency == minimum)
                    .map(|(id, _)| id)
                    .max_by_key(|id| {
                        self.shuffled[..trainset_size]
                            .iter()
                            .position(|candidate| candidate == id)
                            .unwrap_or_default()
                    })
                    .unwrap_or_default();
                self.shuffled.push(selected);
                frequencies[selected] += 1;
            }
        }
        let start = base % self.shuffled.len();
        Ok(self.shuffled[start..start + self.minibatch_size].to_vec())
    }
}

/// CPython `_random.Random` MT19937 stream, including its integer seeding and
/// getrandbits-based `_randbelow` decisions.
#[derive(Debug, Clone)]
struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    fn new(seed: i64) -> Self {
        let magnitude = seed.unsigned_abs();
        let mut key = vec![magnitude as u32];
        if magnitude > u32::MAX as u64 {
            key.push((magnitude >> 32) as u32);
        }
        let mut random = Self {
            state: [0; 624],
            index: 624,
        };
        random.init_by_array(&key);
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..624 {
            self.state[index] = 1_812_433_253_u32
                .wrapping_mul(self.state[index - 1] ^ (self.state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i = 1;
        let mut j = 0;
        for _ in 0..624.max(key.len()) {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn gen_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn getrandbits(&mut self, bits: u32) -> u64 {
        if bits == 0 {
            return 0;
        }
        if bits <= 32 {
            return u64::from(self.gen_u32() >> (32 - bits));
        }
        let low = u64::from(self.gen_u32());
        let remaining = bits - 32;
        let high = u64::from(self.gen_u32() >> (32 - remaining));
        low | (high << 32)
    }

    fn randbelow(&mut self, upper: usize) -> usize {
        let bits = usize::BITS - upper.leading_zeros();
        loop {
            let value = self.getrandbits(bits) as usize;
            if value < upper {
                return value;
            }
        }
    }

    fn random(&mut self) -> f64 {
        let high = self.gen_u32() >> 5;
        let low = self.gen_u32() >> 6;
        (f64::from(high) * 67_108_864.0 + f64::from(low)) / 9_007_199_254_740_992.0
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let selected = self.randbelow(index + 1);
            values.swap(index, selected);
        }
    }
}

fn validate_candidate(candidate: &Candidate) -> Result<(), EngineError> {
    if candidate.is_empty() {
        return Err(EngineError::InvalidParams(
            "seed_candidate must contain at least one component text.".to_string(),
        ));
    }
    for (name, value) in candidate {
        if !value.is_string() {
            return Err(EngineError::InvalidParams(format!(
                "Prompt '{name}' must be a string, got {}",
                python_type_name(value)
            )));
        }
    }
    Ok(())
}

fn candidate_text<'a>(candidate: &'a Candidate, name: &str) -> Result<&'a str, EngineError> {
    candidate
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::InvalidParams(format!("{name} missing in candidate")))
}

fn template_variables(candidate: &Candidate) -> Vec<(String, BTreeSet<String>)> {
    let regex = regex::Regex::new(r"\{\{(\w+)\}\}").expect("static regex is valid");
    candidate
        .iter()
        .map(|(name, value)| {
            let variables = value
                .as_str()
                .into_iter()
                .flat_map(|template| regex.captures_iter(template))
                .map(|capture| capture[1].to_string())
                .collect();
            (name.clone(), variables)
        })
        .collect()
}

fn validate_prompt_names(original: &Candidate, improved: &Candidate) -> Result<(), EngineError> {
    let original_names = original.keys().collect::<BTreeSet<_>>();
    let improved_names = improved.keys().collect::<BTreeSet<_>>();
    let unexpected = improved_names
        .difference(&original_names)
        .map(|name| (*name).clone())
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(EngineError::InvalidParams(format!(
            "Unexpected prompts found in improved prompts: {}",
            python_list(&unexpected)
        )));
    }
    let missing = original_names
        .difference(&improved_names)
        .map(|name| (*name).clone())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(EngineError::InvalidParams(format!(
            "Prompts missing from improved prompts: {}",
            python_list(&missing)
        )));
    }
    Ok(())
}

fn validate_template_variables(
    original: &Candidate,
    improved: &mut Candidate,
) -> Result<(), EngineError> {
    let original_variables = template_variables(original)
        .into_iter()
        .collect::<HashMap<_, _>>();
    let new_variables = template_variables(improved)
        .into_iter()
        .collect::<HashMap<_, _>>();
    for name in original.keys() {
        let missing = original_variables[name]
            .difference(&new_variables[name])
            .cloned()
            .collect::<BTreeSet<_>>();
        if !missing.is_empty() {
            return Err(EngineError::InvalidParams(format!(
                "Template variables mismatch in prompt '{name}'. Missing: {}.",
                python_set(&missing)
            )));
        }
        let extra = new_variables[name]
            .difference(&original_variables[name])
            .cloned()
            .collect::<Vec<_>>();
        if !extra.is_empty() {
            let mut text = candidate_text(improved, name)?.to_string();
            for variable in extra {
                text = text.replace(&format!("{{{{{variable}}}}}"), "");
            }
            improved.insert(name.clone(), Value::String(text));
        }
    }
    Ok(())
}

fn build_meta_prompt(
    config: &MetaPromptConfig,
    prompts: &Candidate,
    variables: &[(String, BTreeSet<String>)],
    evaluations: Option<&[EvaluationRecord]>,
) -> Result<String, EngineError> {
    let current = prompts
        .iter()
        .map(|(name, template)| {
            Ok(format!(
                "Prompt name: {name}\nTemplate: {}",
                template
                    .as_str()
                    .ok_or_else(|| EngineError::InvalidParams(format!(
                        "Prompt '{name}' must be a string"
                    )))?
            ))
        })
        .collect::<Result<Vec<_>, EngineError>>()?
        .join("\n\n");
    let variables = variables
        .iter()
        .map(|(name, values)| {
            format!(
                "- Prompt '{name}': {}",
                if values.is_empty() {
                    "none".to_string()
                } else {
                    values.iter().cloned().collect::<Vec<_>>().join(", ")
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let response_example = prompts
        .keys()
        .map(|name| {
            format!("  \"{name}\": \"improved prompt text with variables preserved exactly\"")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let guidelines = config
        .guidelines
        .as_ref()
        .map(|guidelines| format!("CUSTOM GUIDELINES:\n{guidelines}"))
        .unwrap_or_default();
    let (examples, extra) = match evaluations {
        None => (String::new(), String::new()),
        Some(evaluations) => {
            if evaluations.is_empty() {
                return Err(EngineError::InvalidParams(
                    "Few-shot metaprompting requires evaluation results. No evaluation results were provided to _build_few_shot_meta_prompt."
                        .to_string(),
                ));
            }
            let score = aggregate_score(evaluations);
            let score_info = score
                .map(|score| format!(" (Current Score: {score:.3})"))
                .unwrap_or_default();
            let analysis = if score.is_some() {
                "\nBefore applying best practices, analyze the examples to identify:\n1. **Common Failure Patterns**: What mistakes appear repeatedly? (wrong format,\n   missing steps, calculation errors, etc.)\n2. **Success Patterns**: What made successful examples work? (format, detail level,\n   reasoning approach)\n3. **Key Insights**: What do the rationales tell you about quality criteria and\n   needed improvements?\n4. **Task Requirements**: What output format, explanation level, and edge cases\n   are expected?"
            } else {
                "\nBefore applying best practices, analyze the examples to identify:\n1. **Output Patterns**: What are the expected outputs for different inputs?\n2. **Task Requirements**: What output format, explanation level, and edge cases\n   are expected?\n3. **Common Themes**: What patterns do you see in the input-output relationships?"
            };
            (
                format!(
                    "EVALUATION EXAMPLES{score_info}:\nBelow are examples showing how the current prompts performed. Study these to identify\npatterns in what worked and what failed.\n\n{}\n{analysis}",
                    format_examples(evaluations)
                ),
                "\nFocus on applying best practices that directly address the observed patterns.\nAdd specific instructions, format specifications, or verification steps that would\nimprove the prompt's effectiveness."
                    .to_string(),
            )
        }
    };
    let template = META_PROMPT_TEMPLATE
        .replace("{{", "\u{1}OPEN_BRACE\u{1}")
        .replace("}}", "\u{1}CLOSE_BRACE\u{1}");
    Ok(template
        .replace("{current_prompts_formatted}", &current)
        .replace("{evaluation_examples}", &examples)
        .replace("{template_variables}", &variables)
        .replace("{custom_guidelines}", &guidelines)
        .replace("{extra_instructions}", &extra)
        .replace("{response_format_example}", &response_example)
        .replace("\u{1}OPEN_BRACE\u{1}", "{")
        .replace("\u{1}CLOSE_BRACE\u{1}", "}"))
}

fn format_examples(records: &[EvaluationRecord]) -> String {
    records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let mut lines = vec![
                format!("Example {}:", index + 1),
                format!("  Input: {}", python_json_dumps(&record.inputs)),
                format!("  Output: {}", python_str(&record.outputs)),
                format!("  Expected: {}", python_str(&record.expectations)),
            ];
            if let Some(score) = record.score {
                lines.push(format!("  Score: {score:.3}"));
                let rationales = if record.rationales.is_empty() {
                    "  None".to_string()
                } else {
                    record
                        .rationales
                        .iter()
                        .map(|(name, value)| format!("  - {name}: {}", python_str(value)))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                lines.push(format!("  Rationales:\n{rationales}"));
            }
            format!("{}\n", lines.join("\n"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_metaprompt_response(content: &str) -> Result<Candidate, EngineError> {
    let mut content = content.trim();
    if let Some(rest) = content.strip_prefix("```json") {
        content = rest;
    } else if let Some(rest) = content.strip_prefix("```") {
        content = rest;
    }
    content = content.strip_suffix("```").unwrap_or(content).trim();
    let value: Value = serde_json::from_str(content).map_err(|error| {
        let preview = if content.is_empty() {
            "No content received".to_string()
        } else {
            content.chars().take(2000).collect()
        };
        EngineError::InvalidParams(format!(
            "Failed to parse reflection model response as JSON: {error}\nResponse: {preview}"
        ))
    })?;
    let candidate = value.as_object().cloned().ok_or_else(|| {
        EngineError::InvalidParams(format!(
            "Reflection model returned invalid format. Expected JSON object, got {}",
            python_type_name(&value)
        ))
    })?;
    validate_candidate(&candidate)?;
    Ok(candidate)
}

fn aggregate_score(records: &[EvaluationRecord]) -> Option<f64> {
    if records.is_empty() || records.iter().any(|record| record.score.is_none()) {
        None
    } else {
        Some(
            records
                .iter()
                .filter_map(|record| record.score)
                .sum::<f64>()
                / records.len() as f64,
        )
    }
}

fn per_scorer_scores(records: &[EvaluationRecord]) -> Result<Map<String, Value>, EngineError> {
    let Some(first) = records.first() else {
        return Ok(Map::new());
    };
    let mut result = Map::new();
    for name in first.individual_scores.keys() {
        let scores = records
            .iter()
            .map(|record| {
                record
                    .individual_scores
                    .get(name)
                    .and_then(Value::as_f64)
                    .ok_or_else(|| {
                        EngineError::InvalidParams(format!(
                            "Missing individual score '{name}' in evaluation result"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        result.insert(name.clone(), json!(mean(&scores)));
    }
    Ok(result)
}

fn required_scores(records: &[EvaluationRecord]) -> Result<Vec<f64>, EngineError> {
    records
        .iter()
        .map(|record| {
            record.score.ok_or_else(|| {
                EngineError::InvalidParams(
                    "GEPA evaluation produced a None score; scorers are required".to_string(),
                )
            })
        })
        .collect()
}

fn objective_scores(records: &[EvaluationRecord]) -> Vec<Map<String, Value>> {
    records
        .iter()
        .map(|record| record.individual_scores.clone())
        .collect()
}

fn average_objectives(records: &[Map<String, Value>]) -> Result<Map<String, Value>, EngineError> {
    let mut totals = BTreeMap::<String, (f64, usize)>::new();
    for record in records {
        for (name, value) in record {
            let score = value.as_f64().ok_or_else(|| {
                EngineError::InvalidParams(format!("objective score '{name}' is not numeric"))
            })?;
            let entry = totals.entry(name.clone()).or_default();
            entry.0 += score;
            entry.1 += 1;
        }
    }
    Ok(totals
        .into_iter()
        .map(|(name, (total, count))| (name, json!(total / count as f64)))
        .collect())
}

fn reflective_dataset(
    candidate: &Candidate,
    evaluations: &[EvaluationRecord],
    components: &[String],
) -> Map<String, Value> {
    components
        .iter()
        .map(|component| {
            let records = evaluations
                .iter()
                .enumerate()
                .map(|(index, record)| {
                    json!({
                        "component_name": component,
                        "current_text": candidate.get(component).and_then(Value::as_str).unwrap_or(""),
                        "trace": record.trace_spans,
                        "score": record.score,
                        "inputs": record.inputs,
                        "outputs": record.outputs,
                        "expectations": record.expectations,
                        "rationales": record.rationales,
                        "index": index,
                    })
                })
                .collect();
            (component.clone(), Value::Array(records))
        })
        .collect()
}

fn render_gepa_reflection(current: &str, records: &[Value], template: Option<&str>) -> String {
    let samples = records
        .iter()
        .enumerate()
        .map(|(index, record)| format!("# Example {}\n{}", index + 1, markdown_record(record, 2)))
        .collect::<Vec<_>>()
        .join("\n\n");
    template
        .unwrap_or(GEPA_REFLECTION_TEMPLATE)
        .replace("<curr_instructions>", current)
        .replace("<inputs_outputs_feedback>", &samples)
}

fn markdown_record(value: &Value, level: usize) -> String {
    match value {
        Value::Object(values) => values
            .iter()
            .map(|(name, value)| {
                format!(
                    "{} {name}\n{}",
                    "#".repeat(level),
                    markdown_record(value, (level + 1).min(6))
                )
            })
            .collect(),
        Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                format!(
                    "{} Item {}\n{}",
                    "#".repeat(level),
                    index + 1,
                    markdown_record(value, (level + 1).min(6))
                )
            })
            .collect(),
        Value::Null => "None\n\n".to_string(),
        Value::Bool(value) => format!("{}\n\n", if *value { "True" } else { "False" }),
        Value::Number(value) => format!("{value}\n\n"),
        Value::String(value) => format!("{}\n\n", value.trim()),
    }
}

fn extract_instruction(output: &str) -> String {
    let trimmed = output.trim();
    let first = output.find("```").map(|index| index + 3);
    let last = output.rfind("```");
    match (first, last) {
        (Some(start), Some(end)) if start < end => {
            let content = &output[start..end];
            let content = content
                .split_once('\n')
                .filter(|(language, _)| !language.chars().any(char::is_whitespace))
                .map_or(content, |(_, content)| content);
            content.trim().to_string()
        }
        _ if trimmed.starts_with("```") => trimmed
            .trim_start_matches("```")
            .split_once('\n')
            .map_or_else(
                || trimmed.trim_start_matches("```").trim(),
                |(_, rest)| rest.trim(),
            )
            .to_string(),
        _ if trimmed.ends_with("```") => trimmed[..trimmed.len() - 3].trim().to_string(),
        _ => trimmed.to_string(),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn string_option(
    raw: &Map<String, Value>,
    name: &str,
    default: &str,
) -> Result<String, EngineError> {
    match raw.get(name) {
        None | Some(Value::Null) => Ok(default.to_string()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(value) => Err(type_error(name, "str", value)),
    }
}

fn optional_string(raw: &Map<String, Value>, name: &str) -> Result<Option<String>, EngineError> {
    match raw.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(value) => Err(type_error(name, "str", value)),
    }
}

fn bool_option(raw: &Map<String, Value>, name: &str, default: bool) -> Result<bool, EngineError> {
    match raw.get(name) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(value) => Err(type_error(name, "bool", value)),
    }
}

fn usize_option(
    raw: &Map<String, Value>,
    name: &str,
    default: usize,
) -> Result<usize, EngineError> {
    match raw.get(name) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(value)) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| type_error(name, "int", raw.get(name).expect("present"))),
        Some(value) => Err(type_error(name, "int", value)),
    }
}

fn integer_option(raw: &Map<String, Value>, name: &str, default: i64) -> Result<i64, EngineError> {
    match raw.get(name) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(value)) => value
            .as_i64()
            .ok_or_else(|| type_error(name, "int", raw.get(name).expect("present"))),
        Some(value) => Err(type_error(name, "int", value)),
    }
}

fn number_option(raw: &Map<String, Value>, name: &str, default: f64) -> Result<f64, EngineError> {
    match raw.get(name) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(value)) => value
            .as_f64()
            .ok_or_else(|| type_error(name, "float", raw.get(name).expect("present"))),
        Some(value) => Err(type_error(name, "float", value)),
    }
}

fn type_error(name: &str, expected: &str, value: &Value) -> EngineError {
    EngineError::InvalidParams(format!(
        "{name} must be {expected}, got {}",
        python_type_name(value)
    ))
}

fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_i64() || number.is_u64() => "int",
        Value::Number(_) => "float",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn python_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_string(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => python_repr(value),
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_repr(&Value::String(key.clone())),
                    python_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_json_dumps(value: &Value) -> String {
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_json_dumps)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).expect("string serializes"),
                    python_json_dumps(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => serde_json::to_string(value).expect("JSON value serializes"),
    }
}

fn python_list(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("'{value}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn python_set(values: &BTreeSet<String>) -> String {
    format!(
        "{{{}}}",
        values
            .iter()
            .map(|value| format!("'{value}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

// Worker/store integration follows. It stays in this module so issue-discovery
// work can evolve its own job/store surface independently.

#[derive(Debug, Deserialize)]
struct OptimizeJobParams {
    run_id: String,
    experiment_id: String,
    prompt_uri: String,
    #[serde(default)]
    dataset_id: String,
    optimizer_type: String,
    #[serde(default)]
    optimizer_config: Option<Value>,
    #[serde(default)]
    scorer_names: Vec<String>,
}

#[derive(Clone)]
struct OptimizationClient {
    base: String,
    client: reqwest::Client,
    workspace: Option<String>,
    username: Option<String>,
    password: Option<String>,
}

impl OptimizationClient {
    fn from_request(request: &WorkerRequest) -> Result<Self, EngineError> {
        Ok(Self {
            base: std::env::var("MLFLOW_TRACKING_URI")
                .map_err(|_| {
                    EngineError::Store(
                        "MLFLOW_TRACKING_URI is required for native job execution".to_string(),
                    )
                })?
                .trim_end_matches('/')
                .to_string(),
            client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .map_err(|error| EngineError::Store(error.to_string()))?,
            workspace: request.workspace.clone(),
            username: std::env::var("MLFLOW_TRACKING_USERNAME").ok(),
            password: std::env::var("MLFLOW_TRACKING_PASSWORD").ok(),
        })
    }

    async fn json(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, EngineError> {
        let mut request = self.client.request(method, format!("{}{path}", self.base));
        if let Some(workspace) = &self.workspace {
            request = request.header("X-MLFLOW-WORKSPACE", workspace);
        }
        if let Some(username) = &self.username {
            request = request.basic_auth(username, self.password.as_ref());
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request
            .send()
            .await
            .map_err(|error| EngineError::Store(error.to_string()))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| EngineError::Store(error.to_string()))?;
        if !status.is_success() {
            return Err(EngineError::Store(format!(
                "HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
            )));
        }
        if bytes.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_slice(&bytes).map_err(|error| EngineError::Store(error.to_string()))
    }

    async fn raw(&self, path: &str, bytes: Vec<u8>) -> Result<(), EngineError> {
        let mut request = self.client.post(format!("{}{path}", self.base)).body(bytes);
        if let Some(workspace) = &self.workspace {
            request = request.header("X-MLFLOW-WORKSPACE", workspace);
        }
        if let Some(username) = &self.username {
            request = request.basic_auth(username, self.password.as_ref());
        }
        let response = request
            .send()
            .await
            .map_err(|error| EngineError::Store(error.to_string()))?;
        if !response.status().is_success() {
            return Err(EngineError::Store(format!(
                "HTTP {}: {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }
        Ok(())
    }

    async fn terminate(&self, run_id: &str, status: &str) -> Result<(), EngineError> {
        self.json(
            Method::POST,
            "/api/2.0/mlflow/runs/update",
            Some(&json!({
                "run_id": run_id,
                "status": status,
                "end_time": Utc::now().timestamp_millis(),
            })),
        )
        .await?;
        Ok(())
    }
}

struct WorkerRuntime {
    client: OptimizationClient,
    gateway_url: String,
    source_name: String,
    source_model: String,
    scorers: Vec<SerializedScorer>,
    full_dataset_size: usize,
    run_id: String,
    validation_iteration: usize,
    track_candidates: bool,
    logged_tables: Vec<Value>,
    executor: ScorerExecutor,
}

#[async_trait]
impl OptimizationRuntime for WorkerRuntime {
    async fn evaluate(
        &mut self,
        candidate: &Candidate,
        data: &[Value],
        capture_traces: bool,
    ) -> Result<Vec<EvaluationRecord>, EngineError> {
        let template = candidate_text(candidate, &self.source_name)?;
        let mut records = Vec::with_capacity(data.len());
        for row in data {
            let inputs = row.get("inputs").cloned().ok_or_else(|| {
                EngineError::InvalidParams(
                    "Record is missing required 'inputs' field or it is empty".to_string(),
                )
            })?;
            let formatted = format_prompt(template, &inputs)?;
            let output = call_completion(
                &self.executor,
                &self.gateway_url,
                &self.source_model,
                &formatted,
                false,
                None,
            )
            .await
            .unwrap_or_else(|error| {
                format!(
                    "Failed to invoke the predict_fn with {}: {error}",
                    python_str(&inputs)
                )
            });
            let output = Value::String(output);
            let expectations = row
                .get("expectations")
                .filter(|value| !is_python_falsey(value))
                .cloned()
                .unwrap_or_else(|| json!({"expected_response": row.get("outputs").cloned().unwrap_or(Value::Null)}));
            let trace_spans = vec![json!({
                "name": "litellm.completion",
                "inputs": {"model": self.source_model, "messages": [{"role": "user", "content": formatted}]},
                "outputs": output,
            })];
            let item = EvalItem {
                inputs: Some(inputs.clone()),
                outputs: Some(output.clone()),
                expectations: Some(expectations.clone()),
                trace: Some(json!({"data": {"spans": trace_spans}})),
                session: None,
                memory_examples: None,
            };
            let mut individual = Map::new();
            let mut rationales = Map::new();
            for scorer in &self.scorers {
                let feedback = self
                    .executor
                    .execute_all(scorer, &item, Some(&self.gateway_url), None)
                    .await?;
                let score_value = if feedback.len() == 1 {
                    feedback[0].value.clone()
                } else {
                    Value::Array(feedback.iter().map(|value| value.value.clone()).collect())
                };
                if let Some(score) = numeric_feedback(&score_value) {
                    individual.insert(scorer.common().name.clone(), json!(score));
                }
                if let Some(feedback) = feedback
                    .first()
                    .filter(|feedback| !feedback.rationale.is_empty())
                {
                    rationales.insert(
                        scorer.common().name.clone(),
                        Value::String(feedback.rationale.clone()),
                    );
                }
            }
            if individual.len() != self.scorers.len() {
                let details = self
                    .scorers
                    .iter()
                    .filter(|scorer| !individual.contains_key(&scorer.common().name))
                    .map(|scorer| format!("{} (type: str)", scorer.common().name))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(EngineError::InvalidParams(format!(
                    "Scorers [{details}] return non-numerical values that cannot be automatically aggregated. Please provide an `objective` function to aggregate these values into a single score for optimization."
                )));
            }
            let score = (!self.scorers.is_empty()).then(|| {
                individual.values().filter_map(Value::as_f64).sum::<f64>() / individual.len() as f64
            });
            records.push(EvaluationRecord {
                inputs,
                outputs: output,
                expectations,
                score,
                rationales,
                individual_scores: individual,
                trace_spans,
            });
        }
        if self.track_candidates && !capture_traces && data.len() == self.full_dataset_size {
            self.log_candidate(candidate, &records).await?;
        }
        Ok(records)
    }

    async fn reflect(
        &mut self,
        model: &str,
        prompt: &str,
        json_mode: bool,
        inference: Option<&Map<String, Value>>,
    ) -> Result<String, EngineError> {
        call_completion(
            &self.executor,
            &self.gateway_url,
            &litellm_model(model)?,
            prompt,
            json_mode,
            inference,
        )
        .await
    }
}

impl WorkerRuntime {
    async fn log_candidate(
        &mut self,
        candidate: &Candidate,
        records: &[EvaluationRecord],
    ) -> Result<(), EngineError> {
        let iteration = self.validation_iteration;
        self.validation_iteration += 1;
        let directory = format!("prompt_candidates/iteration_{iteration}");
        let payloads = candidate_artifact_payloads(records)?;
        let aggregate = aggregate_score(records).unwrap_or(0.0);
        self.upload(&format!("{directory}/eval_results.json"), payloads.table)
            .await?;
        self.upload(&format!("{directory}/scores.json"), payloads.scores)
            .await?;
        for (name, prompt) in candidate {
            self.upload(
                &format!("{directory}/{name}.txt"),
                prompt.as_str().unwrap_or_default().as_bytes().to_vec(),
            )
            .await?;
        }
        let mut metrics = BTreeMap::from([("eval_score".to_string(), aggregate)]);
        for (name, score) in payloads.per_scorer {
            if let Some(score) = score.as_f64() {
                metrics.insert(format!("eval_score.{name}"), score);
            }
        }
        log_metrics(&self.client, &self.run_id, &metrics, iteration as i64).await?;
        self.logged_tables.push(json!({
            "path": format!("{directory}/eval_results.json"),
            "type": "table"
        }));
        set_run_tag(
            &self.client,
            &self.run_id,
            LOGGED_ARTIFACTS_TAG,
            &serde_json::to_string(&self.logged_tables).expect("JSON value serializes"),
        )
        .await
    }

    async fn upload(&self, path: &str, bytes: Vec<u8>) -> Result<(), EngineError> {
        self.client
            .raw(
                &format!(
                    "/ajax-api/2.0/mlflow/upload-artifact?run_uuid={}&path={}",
                    urlencoding(&self.run_id),
                    urlencoding(path)
                ),
                bytes,
            )
            .await
    }
}

struct CandidateArtifactPayloads {
    table: Vec<u8>,
    scores: Vec<u8>,
    per_scorer: Map<String, Value>,
}

fn candidate_artifact_payloads(
    records: &[EvaluationRecord],
) -> Result<CandidateArtifactPayloads, EngineError> {
    let aggregate = aggregate_score(records).unwrap_or(0.0);
    let per_scorer = per_scorer_scores(records)?;
    let mut columns = vec![
        "inputs".to_string(),
        "output".to_string(),
        "expectation".to_string(),
        "aggregate_score".to_string(),
    ];
    // Python builds this list from a set. Its column order is intentionally
    // process-dependent; sorting gives stable bytes while preserving the table schema.
    let scorer_names = records
        .iter()
        .flat_map(|record| record.individual_scores.keys().cloned())
        .collect::<BTreeSet<_>>();
    columns.extend(scorer_names.iter().cloned());
    let data = records
        .iter()
        .map(|record| {
            let mut row = vec![
                record.inputs.clone(),
                record.outputs.clone(),
                record.expectations.clone(),
                record.score.map_or(Value::Null, |score| json!(score)),
            ];
            row.extend(scorer_names.iter().map(|name| {
                record
                    .individual_scores
                    .get(name)
                    .cloned()
                    .unwrap_or(Value::Null)
            }));
            Value::Array(row)
        })
        .collect::<Vec<_>>();
    let table = serde_json::to_vec(&json!({"columns": columns, "data": data}))
        .map_err(|error| EngineError::Serialization(error.to_string()))?;
    let scores = serde_json::to_vec_pretty(&json!({
        "aggregate": aggregate,
        "per_scorer": per_scorer,
    }))
    .map_err(|error| EngineError::Serialization(error.to_string()))?;
    Ok(CandidateArtifactPayloads {
        table,
        scores,
        per_scorer,
    })
}

pub(crate) async fn execute_job(request: &WorkerRequest) -> Result<Value, EngineError> {
    let params: OptimizeJobParams = serde_json::from_value(request.params.clone())
        .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
    let client = OptimizationClient::from_request(request)?;
    execute_job_with_client(&params, client).await
}

async fn execute_job_with_client(
    params: &OptimizeJobParams,
    client: OptimizationClient,
) -> Result<Value, EngineError> {
    let result = execute_job_inner(params, client.clone()).await;
    match result {
        Ok(value) => {
            client.terminate(&params.run_id, "FINISHED").await?;
            Ok(value)
        }
        Err(error) => {
            let _ = client.terminate(&params.run_id, "FAILED").await;
            Err(error)
        }
    }
}

async fn execute_job_inner(
    params: &OptimizeJobParams,
    client: OptimizationClient,
) -> Result<Value, EngineError> {
    let (prompt_name, prompt_version) = parse_prompt_uri(&params.prompt_uri)?;
    let source = client
        .json(
            Method::GET,
            &format!(
                "/api/2.0/mlflow/model-versions/get?name={}&version={}",
                urlencoding(&prompt_name),
                urlencoding(&prompt_version)
            ),
            None,
        )
        .await?;
    let tags = tag_map(source.pointer("/model_version/tags"));
    let template = tags.get(PROMPT_TEXT_TAG).ok_or_else(|| {
        EngineError::Store(format!(
            "Prompt {} omitted its template tag",
            params.prompt_uri
        ))
    })?;
    if tags
        .get(PROMPT_TYPE_TAG)
        .is_some_and(|value| value != "text")
    {
        return Err(EngineError::InvalidParams(
            "Only text prompts can be optimized".to_string(),
        ));
    }
    let model_config = tags
        .get(PROMPT_MODEL_CONFIG_TAG)
        .and_then(|value| serde_json::from_str::<Value>(value).ok());
    let provider = model_config
        .as_ref()
        .and_then(|config| config.get("provider"))
        .and_then(Value::as_str);
    let model_name = model_config
        .as_ref()
        .and_then(|config| config.get("model_name"))
        .and_then(Value::as_str);
    let (Some(provider), Some(model_name)) = (provider, model_name) else {
        return Err(EngineError::InvalidParams(format!(
            "Prompt {} doesn't have a model configuration that sets provider and model_name, which are required for optimization.",
            params.prompt_uri
        )));
    };
    let optimizer = parse_job_optimizer(params)?;
    let train_data = load_dataset(&client, &params.dataset_id).await?;
    let scorers = load_scorers(&client, &params.experiment_id, &params.scorer_names).await?;
    link_prompt(&client, &params.run_id, &prompt_name, &prompt_version).await?;
    let gateway_url = std::env::var("MLFLOW_GATEWAY_URI")
        .map(|base| {
            format!(
                "{}/gateway/mlflow/v1/chat/completions",
                base.trim_end_matches('/')
            )
        })
        .map_err(|_| EngineError::MissingGatewayUrl)?;
    let track_candidates = matches!(&optimizer, OptimizerConfig::Gepa(_));
    let runtime = WorkerRuntime {
        client: client.clone(),
        gateway_url,
        source_name: prompt_name.clone(),
        source_model: format!("{provider}/{model_name}"),
        scorers,
        full_dataset_size: train_data.len(),
        run_id: params.run_id.clone(),
        validation_iteration: 0,
        track_candidates,
        logged_tables: Vec::new(),
        executor: ScorerExecutor::new(),
    };
    let mut target = Candidate::new();
    target.insert(prompt_name.clone(), Value::String(template.clone()));
    let mut engine = OptimizationEngine::new(runtime);
    let output = engine.optimize(&optimizer, &train_data, &target).await?;
    let optimized_template = candidate_text(&output.optimized_prompts, &prompt_name)?;
    let optimized_version = register_prompt_version(
        &client,
        &prompt_name,
        optimized_template,
        tags.get(PROMPT_MODEL_CONFIG_TAG),
    )
    .await?;
    link_prompt(&client, &params.run_id, &prompt_name, &optimized_version).await?;
    let optimizer_name = match optimizer {
        OptimizerConfig::Gepa(_) => "GepaPromptOptimizer",
        OptimizerConfig::MetaPrompt(_) => "MetaPromptOptimizer",
    };
    let mut metrics = BTreeMap::new();
    if let Some(score) = output.initial_eval_score {
        metrics.insert("initial_eval_score".to_string(), score);
    }
    if let Some(score) = output.final_eval_score {
        metrics.insert("final_eval_score".to_string(), score);
    }
    for (name, value) in &output.initial_eval_score_per_scorer {
        if let Some(score) = value.as_f64() {
            metrics.insert(format!("initial_eval_score.{name}"), score);
        }
    }
    for (name, value) in &output.final_eval_score_per_scorer {
        if let Some(score) = value.as_f64() {
            metrics.insert(format!("final_eval_score.{name}"), score);
        }
    }
    log_metrics(&client, &params.run_id, &metrics, 0).await?;
    Ok(json!({
        "run_id": params.run_id,
        "source_prompt_uri": params.prompt_uri,
        "optimized_prompt_uri": format!("prompts:/{prompt_name}/{optimized_version}"),
        "optimizer_name": optimizer_name,
        "initial_eval_score": output.initial_eval_score,
        "final_eval_score": output.final_eval_score,
        "dataset_id": params.dataset_id,
        "scorer_names": params.scorer_names,
    }))
}

fn parse_job_optimizer(params: &OptimizeJobParams) -> Result<OptimizerConfig, EngineError> {
    let optimizer_type = params.optimizer_type.to_ascii_lowercase();
    if optimizer_type.is_empty() {
        return Err(EngineError::InvalidParams(
            "Optimizer type must be specified. Supported types: ['gepa', 'metaprompt']".to_string(),
        ));
    }
    if !matches!(optimizer_type.as_str(), "gepa" | "metaprompt") {
        return Err(EngineError::InvalidParams(format!(
            "Unsupported optimizer type: '{}'. Supported types: ['gepa', 'metaprompt']",
            params.optimizer_type
        )));
    }
    let config = match params.optimizer_config.as_ref() {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(config)) => config.clone(),
        Some(value) => {
            return Err(EngineError::InvalidParams(format!(
                "'{}' object has no attribute 'get'",
                python_type_name(value)
            )))
        }
    };
    let reflection_model = config
        .get("reflection_model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            EngineError::InvalidParams(format!(
                "Missing required optimizer configuration: 'reflection_model' must be specified in optimizer_config for the {} optimizer (e.g., 'openai:/gpt-4o').",
                if params.optimizer_type.eq_ignore_ascii_case("gepa") { "GEPA" } else { "MetaPrompt" }
            ))
        })?
        .to_string();
    let _ = litellm_model(&reflection_model)?;
    match optimizer_type.as_str() {
        "gepa" => {
            let max_metric_calls = config.get("max_metric_calls").map_or(Ok(100), |value| {
                value
                    .as_i64()
                    .ok_or_else(|| type_error("max_metric_calls", "int", value))
            })?;
            let gepa_kwargs = config.get("gepa_kwargs").map_or(Ok(Map::new()), |value| {
                value
                    .as_object()
                    .cloned()
                    .ok_or_else(|| type_error("gepa_kwargs", "dict", value))
            })?;
            Ok(OptimizerConfig::Gepa(GepaConfig {
                reflection_model,
                max_metric_calls,
                gepa_kwargs,
            }))
        }
        "metaprompt" => {
            let lm_kwargs = config
                .get("lm_kwargs")
                .filter(|value| !value.is_null())
                .map_or(Ok(Map::new()), |value| {
                    value.as_object().cloned().ok_or_else(|| {
                        EngineError::InvalidParams("`lm_kwargs` must be a dictionary".to_string())
                    })
                })?;
            let guidelines = optional_string(&config, "guidelines")?;
            Ok(OptimizerConfig::MetaPrompt(MetaPromptConfig {
                reflection_model,
                lm_kwargs,
                guidelines,
            }))
        }
        _ => unreachable!("optimizer type was validated above"),
    }
}

async fn load_dataset(
    client: &OptimizationClient,
    dataset_id: &str,
) -> Result<Vec<Value>, EngineError> {
    if dataset_id.is_empty() {
        return Ok(Vec::new());
    }
    let mut records = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let token_query = token
            .as_ref()
            .map(|value| format!("&page_token={}", urlencoding(value)))
            .unwrap_or_default();
        let response = client
            .json(
                Method::GET,
                &format!(
                    "/api/3.0/mlflow/datasets/{}/records?max_results=1000{token_query}",
                    urlencoding(dataset_id)
                ),
                None,
            )
            .await?;
        let encoded = response
            .get("records")
            .and_then(Value::as_str)
            .unwrap_or("[]");
        let page: Vec<Value> =
            serde_json::from_str(encoded).map_err(|error| EngineError::Store(error.to_string()))?;
        records.extend(page);
        token = response
            .get("next_page_token")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        if token.is_none() {
            break;
        }
    }
    for (index, record) in records.iter().enumerate() {
        if record.get("inputs").is_none_or(is_python_falsey) {
            return Err(EngineError::InvalidParams(format!(
                "Record {index} is missing required 'inputs' field or it is empty"
            )));
        }
    }
    Ok(records)
}

async fn load_scorers(
    client: &OptimizationClient,
    experiment_id: &str,
    names: &[String],
) -> Result<Vec<SerializedScorer>, EngineError> {
    let mut scorers = Vec::new();
    for class_name in names {
        if supported_builtin_scorers().contains(&class_name.as_str()) {
            if let Ok(scorer) = builtin_scorer(class_name) {
                scorers.push(scorer);
                continue;
            }
        }
        let response = client
            .json(
                Method::GET,
                &format!(
                    "/api/3.0/mlflow/scorers/get?experiment_id={}&name={}",
                    urlencoding(experiment_id),
                    urlencoding(class_name)
                ),
                None,
            )
            .await
            .map_err(|error| {
                EngineError::InvalidParams(format!(
                    "Scorer '{class_name}' not found. It is neither a built-in scorer (e.g., 'Correctness', 'Safety') nor a registered scorer in experiment '{experiment_id}'. Error: {error}"
                ))
            })?;
        let serialized = response
            .pointer("/scorer/serialized_scorer")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                EngineError::Store("get scorer response omitted serialized_scorer".to_string())
            })?;
        scorers.push(SerializedScorer::from_json(serialized)?);
    }
    Ok(scorers)
}

fn builtin_scorer(class_name: &str) -> Result<SerializedScorer, EngineError> {
    if matches!(
        class_name,
        "ConversationalGuidelines" | "Guidelines" | "RegexMatch" | "ResponseLength"
    ) {
        return Err(EngineError::InvalidParams(format!(
            "{class_name} requires constructor arguments"
        )));
    }
    let name = match class_name {
        "PIIDetection" => "pii_detection".to_string(),
        _ => camel_to_snake(class_name),
    };
    let instructions = match class_name {
        "Correctness" => "Consider the following question, claim and document. You must determine whether the claim is supported by the document in the context of the question. Do not focus on the correctness or completeness of the claim. Do not make assumptions, approximations, or bring in external knowledge.\n\n<question>{{input}}</question>\n<claim>{{ground_truth}}</claim>\n<document>{{input}} - {{output}}</document>",
        "Safety" => "Ensures responses do not contain harmful, offensive, or toxic content.",
        "Equivalence" => "Compare the following actual output against the expected output. You must determine whether they are semantically equivalent or convey the same meaning, and if the output format matches the expected format (e.g., JSON structure, list format, sentence structure).\n\n<actual_output>{{output}}</actual_output>\n<expected_output>{{expected_output}}</expected_output>",
        "RelevanceToQuery" => "Consider the following question and answer. You must determine whether the answer provides information that is (fully or partially) relevant to the question. Do not focus on the correctness or completeness of the answer. Do not make assumptions, approximations, or bring in external knowledge.\n\n<question>{{input}}</question>\n<answer>{{output}}</answer>",
        _ => "",
    };
    SerializedScorer::from_value(json!({
        "name": name,
        "aggregations": null,
        "description": null,
        "is_session_level_scorer": class_name.starts_with("Conversation") || class_name == "UserFrustration",
        "mlflow_version": PINNED_MLFLOW_VERSION,
        "serialization_version": 1,
        "builtin_scorer_class": class_name,
        "builtin_scorer_pydantic_data": {
            "name": name,
            "aggregations": null,
            "description": null,
            "inference_params": null,
            "model": null,
            "instructions": instructions,
        },
    }))
    .map_err(EngineError::from)
}

async fn call_completion(
    executor: &ScorerExecutor,
    gateway_url: &str,
    model: &str,
    prompt: &str,
    json_mode: bool,
    inference: Option<&Map<String, Value>>,
) -> Result<String, EngineError> {
    let mut request = Map::from_iter([
        ("model".to_string(), Value::String(model.to_string())),
        (
            "messages".to_string(),
            json!([{"role": "user", "content": prompt}]),
        ),
    ]);
    if json_mode {
        request.insert(
            "response_format".to_string(),
            json!({"type": "json_object"}),
        );
    }
    if let Some(inference) = inference {
        request.extend(inference.clone());
    }
    let response = executor
        .client()
        .post(gateway_url)
        .json(&request)
        .send()
        .await
        .map_err(|error| EngineError::Gateway(error.to_string()))?;
    let status = response.status();
    let body: Value = response
        .json()
        .await
        .map_err(|error| EngineError::Gateway(error.to_string()))?;
    if !status.is_success() {
        return Err(EngineError::Gateway(format!("HTTP {status}: {body}")));
    }
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| EngineError::MalformedGatewayResponse(body.to_string()))
}

async fn register_prompt_version(
    client: &OptimizationClient,
    name: &str,
    template: &str,
    model_config: Option<&String>,
) -> Result<String, EngineError> {
    let mut tags = vec![
        json!({"key": IS_PROMPT_TAG, "value": "true"}),
        json!({"key": PROMPT_TYPE_TAG, "value": "text"}),
        json!({"key": PROMPT_TEXT_TAG, "value": template}),
    ];
    if let Some(model_config) = model_config {
        tags.push(json!({"key": PROMPT_MODEL_CONFIG_TAG, "value": model_config}));
    }
    let response = client
        .json(
            Method::POST,
            "/api/2.0/mlflow/model-versions/create",
            Some(&json!({
                "name": name,
                "source": "dummy-source",
                "tags": tags,
            })),
        )
        .await?;
    response
        .pointer("/model_version/version")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            EngineError::Store("create model version response omitted version".to_string())
        })
}

async fn link_prompt(
    client: &OptimizationClient,
    run_id: &str,
    name: &str,
    version: &str,
) -> Result<(), EngineError> {
    let run = client
        .json(
            Method::GET,
            &format!("/api/2.0/mlflow/runs/get?run_id={}", urlencoding(run_id)),
            None,
        )
        .await?;
    let current = run
        .pointer("/run/data/tags")
        .and_then(Value::as_array)
        .and_then(|tags| {
            tags.iter().find_map(|tag| {
                (tag.get("key").and_then(Value::as_str) == Some(LINKED_PROMPTS_TAG))
                    .then(|| tag.get("value").and_then(Value::as_str))
                    .flatten()
            })
        });
    let mut links = current
        .map(serde_json::from_str::<Vec<Value>>)
        .transpose()
        .map_err(|_| {
            EngineError::InvalidParams(format!(
                "Invalid JSON format for '{LINKED_PROMPTS_TAG}' tag: {}",
                current.unwrap_or_default()
            ))
        })?
        .unwrap_or_default();
    let link = json!({"name": name, "version": version});
    if !links.contains(&link) {
        links.push(link);
        set_run_tag(
            client,
            run_id,
            LINKED_PROMPTS_TAG,
            &serde_json::to_string(&links).expect("JSON value serializes"),
        )
        .await?;
    }
    Ok(())
}

async fn set_run_tag(
    client: &OptimizationClient,
    run_id: &str,
    key: &str,
    value: &str,
) -> Result<(), EngineError> {
    client
        .json(
            Method::POST,
            "/api/2.0/mlflow/runs/set-tag",
            Some(&json!({"run_id": run_id, "key": key, "value": value})),
        )
        .await?;
    Ok(())
}

async fn log_metrics(
    client: &OptimizationClient,
    run_id: &str,
    metrics: &BTreeMap<String, f64>,
    step: i64,
) -> Result<(), EngineError> {
    if metrics.is_empty() {
        return Ok(());
    }
    let timestamp = Utc::now().timestamp_millis();
    client
        .json(
            Method::POST,
            "/api/2.0/mlflow/runs/log-batch",
            Some(&json!({
                "run_id": run_id,
                "metrics": metrics.iter().map(|(key, value)| json!({
                    "key": key,
                    "value": value,
                    "timestamp": timestamp,
                    "step": step,
                })).collect::<Vec<_>>(),
                "params": [],
                "tags": [],
            })),
        )
        .await?;
    Ok(())
}

fn parse_prompt_uri(uri: &str) -> Result<(String, String), EngineError> {
    let rest = uri
        .strip_prefix("prompts:/")
        .ok_or_else(|| EngineError::InvalidParams(format!("Invalid prompt URI: {uri}")))?;
    let (name, version) = rest
        .rsplit_once('/')
        .ok_or_else(|| EngineError::InvalidParams(format!("Invalid prompt URI: {uri}")))?;
    if name.is_empty() || version.is_empty() {
        return Err(EngineError::InvalidParams(format!(
            "Invalid prompt URI: {uri}"
        )));
    }
    Ok((name.to_string(), version.to_string()))
}

fn litellm_model(uri: &str) -> Result<String, EngineError> {
    let (provider, model) = uri.split_once(":/").ok_or_else(|| {
        EngineError::InvalidParams(format!(
            "Malformed model uri '{uri}'. The URI must be in the format of <provider>:/<model-name>, e.g., 'openai:/gpt-4.1-mini'."
        ))
    })?;
    let model = model.trim_start_matches('/');
    if provider.is_empty() || model.is_empty() {
        return Err(EngineError::InvalidParams(format!(
            "Malformed model uri '{uri}'. The URI must be in the format of <provider>:/<model-name>, e.g., 'openai:/gpt-4.1-mini'."
        )));
    }
    Ok(format!("{provider}/{model}"))
}

fn tag_map(tags: Option<&Value>) -> HashMap<String, String> {
    tags.and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tag| {
            Some((
                tag.get("key")?.as_str()?.to_string(),
                tag.get("value")?.as_str()?.to_string(),
            ))
        })
        .collect()
}

fn format_prompt(template: &str, inputs: &Value) -> Result<String, EngineError> {
    let inputs = inputs.as_object().ok_or_else(|| {
        EngineError::InvalidParams("Prompt inputs must be a dictionary".to_string())
    })?;
    let mut result = template.to_string();
    for (key, value) in inputs {
        let regex = regex::Regex::new(&format!(r"\{{\{{\s*{}\s*\}}\}}", regex::escape(key)))
            .map_err(|error| EngineError::InvalidParams(error.to_string()))?;
        result = regex
            .replace_all(&result, regex::NoExpand(&python_str(value)))
            .into_owned();
    }
    let variables = template_variables(&Map::from_iter([(
        "prompt".to_string(),
        Value::String(template.to_string()),
    )]));
    let prompt_variables = variables
        .first()
        .map(|(_, variables)| variables)
        .expect("the synthetic prompt entry exists");
    let missing = prompt_variables
        .iter()
        .filter(|name| !inputs.contains_key(*name))
        .cloned()
        .collect::<BTreeSet<_>>();
    if !missing.is_empty() {
        return Err(EngineError::InvalidParams(format!(
            "Missing variables: {}. To partially format the prompt, set `allow_partial=True`.",
            python_set(&missing)
        )));
    }
    Ok(result)
}

fn numeric_feedback(value: &Value) -> Option<f64> {
    match value {
        Value::String(value) if value == "yes" => Some(1.0),
        Value::String(value) if value == "no" => Some(0.0),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Number(value) => value.as_f64(),
        _ => None,
    }
}

fn is_python_falsey(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(value) => !value,
        Value::Number(value) => value.as_f64() == Some(0.0),
        Value::String(value) => value.is_empty(),
        Value::Array(value) => value.is_empty(),
        Value::Object(value) => value.is_empty(),
    }
}

fn camel_to_snake(value: &str) -> String {
    let mut result = String::new();
    for (index, character) in value.chars().enumerate() {
        if character.is_uppercase() && index > 0 {
            result.push('_');
        }
        result.extend(character.to_lowercase());
    }
    result
}

fn urlencoding(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use axum::body::Bytes;
    use axum::extract::{OriginalUri, State};
    use axum::routing::any;
    use axum::{Json, Router};
    use sha2::{Digest, Sha256};

    #[derive(Default)]
    struct ScriptedRuntime {
        reflection_count: usize,
        captured_batches: Vec<Vec<i64>>,
    }

    #[async_trait]
    impl OptimizationRuntime for ScriptedRuntime {
        async fn evaluate(
            &mut self,
            candidate: &Candidate,
            data: &[Value],
            capture_traces: bool,
        ) -> Result<Vec<EvaluationRecord>, EngineError> {
            let text = candidate_text(candidate, "prompt")?;
            let version = text
                .strip_prefix("candidate-")
                .and_then(|value| value.parse::<i64>().ok())
                .unwrap_or_default();
            if capture_traces {
                self.captured_batches
                    .push(data.iter().filter_map(Value::as_i64).collect::<Vec<_>>());
            }
            Ok(data
                .iter()
                .map(|value| {
                    let id = value.as_i64().expect("scripted data is integral");
                    let score = ((id + version) % 5) as f64 / 4.0;
                    EvaluationRecord {
                        inputs: json!({"id": id}),
                        outputs: json!(format!("{version}:{id}")),
                        expectations: json!({"expected_response": id}),
                        score: Some(score),
                        rationales: Map::new(),
                        individual_scores: Map::from_iter([("quality".to_string(), json!(score))]),
                        trace_spans: vec![json!({
                            "name": "scripted",
                            "inputs": {"id": id},
                            "outputs": format!("{version}:{id}"),
                        })],
                    }
                })
                .collect())
        }

        async fn reflect(
            &mut self,
            _model: &str,
            _prompt: &str,
            _json_mode: bool,
            _inference: Option<&Map<String, Value>>,
        ) -> Result<String, EngineError> {
            self.reflection_count += 1;
            Ok(format!("```candidate-{}```", self.reflection_count))
        }
    }

    #[test]
    fn python_rng_matches_cpython_decisions() {
        let mut random = PythonRandom::new(0);
        assert_eq!(
            (0..10).map(|_| random.randbelow(17)).collect::<Vec<_>>(),
            vec![12, 13, 1, 8, 16, 15, 12, 9, 15, 11]
        );
        let mut random = PythonRandom::new(42);
        let mut values = (0..8).collect::<Vec<_>>();
        random.shuffle(&mut values);
        assert_eq!(values, vec![3, 4, 6, 7, 2, 5, 0, 1]);
    }

    #[test]
    fn metaprompt_json_cleanup_and_variable_validation_match_python() {
        let parsed = parse_metaprompt_response(
            "```json\n{\"qa\":\"Answer {{question}} and {{invented}}\"}\n```",
        )
        .unwrap();
        let mut original = Candidate::new();
        original.insert("qa".to_string(), json!("Q: {{question}}"));
        let mut parsed = parsed;
        validate_prompt_names(&original, &parsed).unwrap();
        validate_template_variables(&original, &mut parsed).unwrap();
        assert_eq!(parsed["qa"], "Answer {{question}} and ");
    }

    #[test]
    fn metaprompt_construction_matches_python_bytes() {
        let prompts = Candidate::from_iter([("qa".to_string(), json!("Answer {{question}}."))]);
        let config = MetaPromptConfig {
            reflection_model: "openai:/fake-model".to_string(),
            lm_kwargs: Map::new(),
            guidelines: Some("Prefer concise answers.".to_string()),
        };
        let variables = template_variables(&prompts);
        let zero = build_meta_prompt(&config, &prompts, &variables, None).unwrap();
        assert_eq!(zero.len(), 2805);
        assert_eq!(
            format!("{:x}", Sha256::digest(zero.as_bytes())),
            "594ffc4bb7031773e33809b0ad4cf37b38593015b68a1059f344bd28f7418a6e"
        );
        let evaluations = vec![EvaluationRecord {
            inputs: json!({"question": "2+2?"}),
            outputs: json!("5"),
            expectations: json!({"expected_response": "4"}),
            score: Some(0.0),
            rationales: Map::from_iter([("accuracy".to_string(), json!("Incorrect"))]),
            individual_scores: Map::from_iter([("accuracy".to_string(), json!(0.0))]),
            trace_spans: Vec::new(),
        }];
        let few = build_meta_prompt(&config, &prompts, &variables, Some(&evaluations)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(few.as_bytes())),
            "bc39355534afef48883c5fefdb337f7ff20f3aa73a47e7bec9555b0fa665bb54"
        );
    }

    #[tokio::test]
    async fn gepa_candidate_sequence_and_budget_match_python_0_0_27() {
        let runtime = ScriptedRuntime::default();
        let mut engine = OptimizationEngine::new(runtime);
        let output = engine
            .optimize(
                &OptimizerConfig::Gepa(GepaConfig {
                    reflection_model: "openai:/fake-model".to_string(),
                    max_metric_calls: 35,
                    gepa_kwargs: Map::from_iter([("seed".to_string(), json!(7))]),
                }),
                &[json!(0), json!(1), json!(2), json!(3), json!(4)],
                &Candidate::from_iter([("prompt".to_string(), json!("seed"))]),
            )
            .await
            .unwrap();
        assert_eq!(
            engine.runtime().captured_batches,
            vec![vec![4, 0, 2], vec![3, 1, 1], vec![1, 4, 3], vec![2, 0, 0]]
        );
        assert_eq!(
            output.candidate_sequence,
            vec![
                Candidate::from_iter([("prompt".to_string(), json!("seed"))]),
                Candidate::from_iter([("prompt".to_string(), json!("candidate-2"))]),
                Candidate::from_iter([("prompt".to_string(), json!("candidate-4"))]),
            ]
        );
        assert_eq!(output.selected_candidate_indices, vec![0, 0, 0, 0]);
        assert_eq!(output.validation_scores, vec![0.5, 0.5, 0.5]);
        assert_eq!(output.metric_calls, 39);
    }

    #[test]
    fn candidate_artifact_bytes_match_python() {
        let record = |id, output: &str, expected: &str, score| EvaluationRecord {
            inputs: json!({"id": id}),
            outputs: json!(output),
            expectations: json!({"expected_response": expected}),
            score: Some(score),
            rationales: Map::new(),
            individual_scores: Map::from_iter([("quality".to_string(), json!(score))]),
            trace_spans: Vec::new(),
        };
        let payloads =
            candidate_artifact_payloads(&[record(0, "a", "a", 1.0), record(1, "b", "c", 0.0)])
                .unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&payloads.table)),
            "496625d15c47d9fd2195d5976121855b5645dafa4bcc9151ac4304433f32283a"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&payloads.scores)),
            "353cbf458d0bfd738e0e41a4c7820966fd4457842ce773211b96afa5ba36e966"
        );
    }

    #[derive(Clone, Default)]
    struct RegistryScript {
        requests: Arc<Mutex<Vec<(String, String, Value)>>>,
        links: Arc<Mutex<Vec<Value>>>,
    }

    async fn registry_handler(
        State(script): State<RegistryScript>,
        method: Method,
        OriginalUri(uri): OriginalUri,
        body: Bytes,
    ) -> Json<Value> {
        let payload = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
        script.requests.lock().unwrap().push((
            method.to_string(),
            uri.path().to_string(),
            payload.clone(),
        ));
        let response = match (method, uri.path()) {
            (Method::GET, "/api/2.0/mlflow/model-versions/get") => json!({
                "model_version": {"tags": [
                    {"key": IS_PROMPT_TAG, "value": "true"},
                    {"key": PROMPT_TYPE_TAG, "value": "text"},
                    {"key": PROMPT_TEXT_TAG, "value": "Answer {{question}}"},
                ]}
            }),
            (Method::POST, "/api/2.0/mlflow/model-versions/create") => {
                json!({"model_version": {"version": "2"}})
            }
            (Method::GET, "/api/2.0/mlflow/runs/get") => json!({
                "run": {"data": {"tags": [{
                    "key": LINKED_PROMPTS_TAG,
                    "value": serde_json::to_string(&*script.links.lock().unwrap()).unwrap(),
                }]}}
            }),
            (Method::POST, "/api/2.0/mlflow/runs/set-tag") => {
                if payload["key"] == LINKED_PROMPTS_TAG {
                    *script.links.lock().unwrap() =
                        serde_json::from_str(payload["value"].as_str().unwrap()).unwrap();
                }
                json!({})
            }
            _ => json!({}),
        };
        Json(response)
    }

    #[tokio::test]
    async fn prompt_registration_and_run_linkage_match_python() {
        let script = RegistryScript::default();
        script
            .links
            .lock()
            .unwrap()
            .push(json!({"name": "qa", "version": "1"}));
        let app = Router::new()
            .fallback(any(registry_handler))
            .with_state(script.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = OptimizationClient {
            base: format!("http://{address}"),
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            workspace: Some("workspace-a".to_string()),
            username: None,
            password: None,
        };
        let model_config = json!({"provider": "openai", "model_name": "fake"}).to_string();
        let version =
            register_prompt_version(&client, "qa", "Improved {{question}}", Some(&model_config))
                .await
                .unwrap();
        assert_eq!(version, "2");
        link_prompt(&client, "run-1", "qa", &version).await.unwrap();
        assert_eq!(
            *script.links.lock().unwrap(),
            vec![
                json!({"name": "qa", "version": "1"}),
                json!({"name": "qa", "version": "2"}),
            ]
        );
        let requests = script.requests.lock().unwrap();
        let create = requests
            .iter()
            .find(|(_, path, _)| path == "/api/2.0/mlflow/model-versions/create")
            .unwrap();
        assert_eq!(create.2["name"], "qa");
        assert_eq!(create.2["source"], "dummy-source");
        let tags = tag_map(create.2.get("tags"));
        assert_eq!(tags[IS_PROMPT_TAG], "true");
        assert_eq!(tags[PROMPT_TYPE_TAG], "text");
        assert_eq!(tags[PROMPT_TEXT_TAG], "Improved {{question}}");
        assert_eq!(tags[PROMPT_MODEL_CONFIG_TAG], model_config);
    }

    struct FailureRuntime {
        fail_evaluation: bool,
    }

    #[async_trait]
    impl OptimizationRuntime for FailureRuntime {
        async fn evaluate(
            &mut self,
            _candidate: &Candidate,
            data: &[Value],
            _capture_traces: bool,
        ) -> Result<Vec<EvaluationRecord>, EngineError> {
            if self.fail_evaluation {
                return Err(EngineError::Gateway("scripted scorer failure".to_string()));
            }
            Ok(data
                .iter()
                .map(|input| EvaluationRecord {
                    inputs: input.clone(),
                    outputs: json!("wrong"),
                    expectations: json!({"expected_response": "right"}),
                    score: Some(0.25),
                    rationales: Map::new(),
                    individual_scores: Map::from_iter([("quality".to_string(), json!(0.25))]),
                    trace_spans: vec![json!({"name": "scripted"})],
                })
                .collect())
        }

        async fn reflect(
            &mut self,
            _model: &str,
            _prompt: &str,
            _json_mode: bool,
            _inference: Option<&Map<String, Value>>,
        ) -> Result<String, EngineError> {
            Err(EngineError::Gateway(
                "scripted reflection failure".to_string(),
            ))
        }
    }

    #[tokio::test]
    async fn scripted_failure_semantics_match_python() {
        let prompts = Candidate::from_iter([("qa".to_string(), json!("Answer {{question}}"))]);
        let mut metaprompt = OptimizationEngine::new(FailureRuntime {
            fail_evaluation: false,
        });
        let output = metaprompt
            .optimize(
                &OptimizerConfig::MetaPrompt(MetaPromptConfig {
                    reflection_model: "openai:/fake-model".to_string(),
                    lm_kwargs: Map::new(),
                    guidelines: None,
                }),
                &[json!({"question": "test"})],
                &prompts,
            )
            .await
            .unwrap();
        assert_eq!(output.optimized_prompts, prompts);
        assert_eq!(output.initial_eval_score, Some(0.25));
        assert_eq!(output.final_eval_score, None);

        let mut gepa = OptimizationEngine::new(FailureRuntime {
            fail_evaluation: true,
        });
        let error = gepa
            .optimize(
                &OptimizerConfig::Gepa(GepaConfig {
                    reflection_model: "openai:/fake-model".to_string(),
                    max_metric_calls: 10,
                    gepa_kwargs: Map::new(),
                }),
                &[json!({"question": "test"})],
                &prompts,
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "gateway request failed: scripted scorer failure"
        );
    }

    #[tokio::test]
    async fn failed_job_marks_run_failed_with_python_message() {
        let script = RegistryScript::default();
        let app = Router::new()
            .fallback(any(registry_handler))
            .with_state(script.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = OptimizationClient {
            base: format!("http://{address}"),
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            workspace: None,
            username: None,
            password: None,
        };
        let params = OptimizeJobParams {
            run_id: "run-failure".to_string(),
            experiment_id: "1".to_string(),
            prompt_uri: "prompts:/qa/1".to_string(),
            dataset_id: String::new(),
            optimizer_type: "metaprompt".to_string(),
            optimizer_config: Some(json!({"reflection_model": "openai:/fake-model"})),
            scorer_names: Vec::new(),
        };
        let error = execute_job_with_client(&params, client).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid invoke_scorer parameters: Prompt prompts:/qa/1 doesn't have a model configuration that sets provider and model_name, which are required for optimization."
        );
        let requests = script.requests.lock().unwrap();
        let update = requests
            .iter()
            .find(|(_, path, _)| path == "/api/2.0/mlflow/runs/update")
            .unwrap();
        assert_eq!(update.2["run_id"], "run-failure");
        assert_eq!(update.2["status"], "FAILED");
        assert!(update.2["end_time"].as_i64().is_some());
    }
}
