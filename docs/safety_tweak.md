# Safety Tweak guide

Safety Tweak localizes the MLP neurons that are most differentially active on
refusal-triggering prompts and suppresses them. It is training-free and
reversible: the original model directory is never modified, and edits are
applied from the original weights on every run.

## Pipeline

1. **Load prompts.** The safety dataset has 600 harmless prompts and 600
   refusal-triggering prompts. `--evaluation-size` (default 100) holds out the
   same number from each class. The remaining 500 per class are used for
   localization.
2. **Capture activations.** The tool runs the localization prompts through the
   model and records the input to each MLP `down_proj`. That input has shape
   `(batch, sequence, intermediate_size)`; each column is one MLP neuron. The
   capture stores the sum, sum of squares, and token count per neuron per layer.
3. **Score neurons.** For each neuron:

   ```
   d = (mean_bad - mean_good) / sqrt((var_good + var_bad) / 2 + eps)
   ```

   This is Cohen's d. Positive values mean the neuron is more active on
   refusal-triggering prompts.
4. **Select neurons.** The top `--top-neurons` per layer (default 10) with a
   score of at least `--min-score` (default 0) are selected. Use `--layers` to
   restrict the analysis, for example `--layers 10-20`.
5. **Build a plan.** Each selected neuron becomes an `L<layer> N<neuron>
   <delta>` entry. The default delta is `-1`, which deletes the neuron's
   contribution. Use `--neuron-delta -0.5` for a gentler suppression, or a
   positive delta to amplify.
6. **Apply and save.** The plan is applied to `--target` (default `down_proj`),
   and the edited model is written to `--save-directory` when provided. With
   `--evaluate`, the tool reports the refusal rate before applying the plan and
   again after applying it.

## Commands

```bash
# Dry run: analyze and print the plan without changing weights.
neuron-tweak <model> --profile safety --dry-run

# Apply the default plan and save.
neuron-tweak <model> --profile safety --yes --save-directory ./out

# Apply, save, and report the refusal rate before/after the edit.
neuron-tweak <model> --profile safety --evaluate --yes --save-directory ./out

# Quick pass with 50 prompts per class.
neuron-tweak <model> --profile safety --analysis-prompts 50 --dry-run

# Middle layers only, five neurons per layer, gentle suppression.
neuron-tweak <model> --profile safety \
  --layers 10-20 --top-neurons 5 --neuron-delta -0.5 \
  --save-directory ./out

# Delete the selected neurons from all three MLP projections.
neuron-tweak <model> --profile safety --target all --save-directory ./out
```

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--analysis-prompts` | `0` | Training prompts per class; `0` uses all 500. |
| `--evaluation-size` | `100` | Held-out prompts per class. |
| `--analysis-batch-size` | `8` | Prompts per forward pass. |
| `--max-sequence-length` | `512` | Maximum tokens per prompt. |
| `--top-neurons` | `10` | Neurons selected per layer. |
| `--min-score` | `0.0` | Minimum Cohen's d. |
| `--layers` | `all` | `all`, `14,15`, or `10-20`. |
| `--neuron-delta` | `-1.0` | Delta written into the plan. |
| `--target` | `down_proj` | `down_proj`, `up_proj`, `gate_proj`, or `all`. |
| `--seed` | `0` | Shuffle seed for the train/evaluation split. |
| `--yes` | off | Skip the apply confirmation. |
| `--dry-run` | off | Analyze and print, but do not change weights. |
| `--evaluate` | off | Generate before/after responses and report the refusal rate. |
| `--eval-max-new-tokens` | `64` | Max new tokens for evaluation responses. |
| `--eval-batch-size` | `4` | Batch size for evaluation responses. |
| `--save-directory` | none | Directory for the edited model. |

## Evaluation

With `--evaluate`, the tool generates responses for the held-out bad prompts
before applying the plan and again after applying it, then reports the refusal
rate (the fraction of responses containing a refusal marker) in both runs:

- Before: reported before the edits are applied.
- After: reported after the edits are applied.
- In dry-run mode, only the before rate is reported and no weights change.

```bash
neuron-tweak <model> --profile safety --evaluate --yes --save-directory ./out
```

## Optuna study

With `--study`, the tool runs an Optuna study over the selected safety neurons.

- Each trial samples `top_k` (from 1 to the number of selected neurons) and a
  steering `delta` (from -1 to 1).
- The trial applies the corresponding plan and measures the held-out refusal
  rate.
- The best trial is reported, and `--study-apply-best` applies and saves it.

```bash
neuron-tweak <model> --profile safety \
  --study --study-trials 30 --study-apply-best --save-directory ./out
```

## What the edit changes

Scaling the `down_proj` column by `1 + delta` changes the neuron's output
contribution. With the default `delta = -1`, the column becomes zero: the
neuron no longer contributes to the residual stream. The up and gate
projections keep their original values, so the edit is minimal and easy to
reverse. `--target all` also scales the matching row of `up_proj` and
`gate_proj`, which removes the neuron from every stage of the MLP.

## Interpretation and limits

- The first version uses a single mean activation per neuron over all tokens.
  This is fast and works well for localization, but it does not separate
  prompt tokens from response tokens. A later phase will capture response-only
  activations and add logit-lens scoring.
- The 100 held-out evaluation prompts are used for automated refusal-rate
  scoring when `--evaluate` is set. Before applying the plan the tool reports the
  refusal rate, applies the edits, and reports the refusal rate again. Until
  that phase lands in the general evaluation flow, compare a few generations
  manually before trusting an edit.
- Cohen's d is a screening signal, not proof of causality. Inspect the top
  neurons, start with a small `--top-neurons` value, and keep the original
  model.
- Very large positive deltas can degrade the model. Negative deltas on a small
  set of high-scoring neurons are the safest starting point.
- Activation capture needs a tokenizer. Models loaded without one are rejected
  with a clear error.
- Full-precision weights are required; quantized models are not supported by
  the current editing runtime.
