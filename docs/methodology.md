# Methodology

Neuron Tweak adapts the Heretic pipeline to neuron-level interventions. The
pipeline has five stages.

## 1. Load and benchmark

The model is loaded locally with transformers and accelerate. Supported dtypes
are tried in order until one succeeds. Optional 4-bit quantization via
bitsandbytes reduces VRAM use at a small accuracy cost. The batch size is
chosen by measuring tokens per second across increasing batch sizes.

## 2. Contrastive activation capture

For every prompt in the good and bad sets, forward hooks record activations for
the layers of interest. For each transformer block this includes the MLP
up-projection, gate-projection, and down-projection outputs, and optionally the
attention output projection. Activations are moved to CPU as soon as possible
to limit peak VRAM use.

## 3. Neuron scoring and localization

For each layer and each neuron, the engine computes a separation score between
the bad and good activation distributions. The default scores are:

- Cohen's d between the means, with optional winsorization to limit the effect
  of massive activations.
- A logistic-probe margin fitted on CPU.
- For identity, a logit-lens projection of the neuron activation onto the
  unembedding directions of current-identity and target-identity tokens.

Identity localization is validated with causal tracing: activations from a
target-identity run are patched into a current-identity run, and the change in
identity-token logits is measured. Only neurons with a positive causal effect
are retained.

## 4. Editing

Safety and identity edits use different operators but share the same reversible
LoRA machinery.

Safety operators:

- Downscale the incoming weights of selected neurons and the matching column of
  the down-projection matrix.
- Clamp the neuron's contribution to zero.
- Shift the pre-activation with a negative bias where the architecture allows
  it.

Identity operator:

- Optimize a low-rank weight edit on the selected neurons so that target
  identity tokens become likely on identity prompts while unrelated factual
  prompts remain close to the original model.

Every edit is expressed relative to the frozen base weights, so reset restores
the original model exactly.

## 5. Optimization and selection

An Optuna TPE study samples the editable parameters: layers, top-k neurons,
scale or strength, LoRA rank, and operator. Each trial is scored by the
configured scorers, and the Pareto front is presented in the terminal. The
user chooses a trial, inspects its scores, and then saves, chats with,
benchmarks, or uploads the resulting model.

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
