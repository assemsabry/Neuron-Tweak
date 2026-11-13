# Methodology

Neuron Tweak adapts the Heretic pipeline to neuron-level interventions. The
pipeline has five stages.

## 1. Load and benchmark

The model is loaded locally with transformers and accelerate. Supported dtypes
are tried in order until one succeeds. The current neuron-editing runtime
requires full-precision weights; 4-bit quantization is rejected for editing.
Batch-size auto-tuning is not implemented yet; the analysis batch size is
configurable with `--analysis-batch-size`.

## 2. Contrastive activation capture

For every prompt in the good and bad sets, forward pre-hooks on each
`mlp.down_proj` record the neuron activations: the input to the down-projection
has shape `(batch, sequence, intermediate_size)`, so each column is one MLP
neuron. The current implementation accumulates the sum, sum of squares, and
token count per neuron per selected layer on CPU to limit peak VRAM use.
Attention, up-projection, and gate-projection capture are planned.

## 3. Neuron scoring and localization

For each layer and each neuron, the engine computes a separation score between
the bad and good activation distributions.

Implemented:

- Cohen's d between the means:
  `(mean_bad - mean_good) / sqrt((var_good + var_bad) / 2 + eps)`.

Planned:

- Winsorization to limit the effect of massive activations.
- A logistic-probe margin fitted on CPU.
- For identity, a logit-lens projection of the neuron activation onto the
  unembedding directions of current-identity and target-identity tokens.
- Causal tracing for identity: activations from a target-identity run are
  patched into a current-identity run, and the change in identity-token logits
  is measured.

## 4. Editing

Safety edits use direct, reversible projection scaling. Identity edits will use
the same reversible machinery with a learned low-rank edit.

Safety operator (implemented):

- Scale the matching `down_proj` column by `clamp(1 + delta, 0, max_scale)`.
  The default `delta = -1` removes the neuron's contribution.
- With `--target up_proj`, `gate_proj`, or `all`, the matching rows are scaled
  as well, removing the neuron from every stage of the MLP.

Identity operator (planned):

- Optimize a low-rank weight edit on the selected neurons so that target
  identity tokens become likely on identity prompts while unrelated factual
  prompts remain close to the original model.

Custom operator:

- The user supplies exact `L<layer> N<neuron> <delta>` entries.
- The tool multiplies the matching MLP projection vector by
  `clamp(1 + delta, 0, max_scale)`. `--target` selects `down_proj` (default),
  `up_proj`, `gate_proj`, or `all`.
- A delta of `-1` deletes the neuron's contribution; `+1` doubles it.
- The editor supports adding, updating, removing, and clearing entries before
  the plan is applied. Applying a new plan starts from the original weights, so
  entries are absolute rather than cumulative.
- Custom Tweak uses no prompt dataset, scoring, or optimizer.

Every edit is expressed relative to the frozen base weights, so reset restores
the original model exactly.

## 5. Optimization and selection

Current Safety Tweak: select the top `--top-neurons` per layer by Cohen's d and
apply a fixed `--neuron-delta` to the chosen projection. This is a transparent
first pass, not a tuned optimum.

Planned: an Optuna TPE study that samples layers, top-k neurons, scale or
strength, LoRA rank, and operator. Each trial is scored by the configured
scorers, and the Pareto front is presented in the terminal. The user chooses a
trial, inspects its scores, and then saves, chats with, benchmarks, or uploads
the resulting model.

## Prompt counts

The project ships with 600 good and 600 bad prompts per workflow. By default
each set is shuffled with a fixed seed, 500 prompts are used for localization
and optimization, and 100 are held out for evaluation. This is intentionally
larger than the 400-prompt contrast sets used by Heretic.

## References

- Arditi et al., "Refusal in Language Models Is Mediated by a Single
  Direction", 2024.
- Lai, "Projected Abliteration", 2025.
- Lai, "Norm-Preserving Biprojected Abliteration", 2025.
- Heretic: https://github.com/p-e-w/heretic
