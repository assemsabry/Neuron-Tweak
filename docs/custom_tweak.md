# Custom Tweak guide

Custom Tweak gives direct control over individual MLP neurons. It does not use
prompt datasets, scoring, or an optimizer. The user supplies a plan and the tool
applies it to the model.

## Neuron address

```
L<layer> N<neuron> <delta>
```

- `L` is 1-based. `L14` is the 14th transformer block.
- `N` is a 0-based index into the MLP intermediate dimension.
- `delta` is a signed number.

Examples:

```
L14 N8949 +1
L13 N7956 -1
```

## What the edit changes

For a Llama-style block, the MLP is:

```
up_proj:   hidden_size -> intermediate_size
gate_proj: hidden_size -> intermediate_size
down_proj: intermediate_size -> hidden_size
```

`down_proj.weight` has shape `(hidden_size, intermediate_size)`. Column `i`
maps neuron `i` back to the hidden dimension. Custom Tweak multiplies that
column by:

```
scale = clamp(1 + delta, 0, max_scale)
```

- `+1` gives `scale = 2.0`, doubling the neuron's contribution.
- `-1` gives `scale = 0.0`, removing the neuron's contribution.
- `-0.5` gives `scale = 0.5`, halving it.
- `+0.5` gives `scale = 1.5`, increasing it by 50 percent.
- `delete` is an alias for `-1`.
- `reset` removes the entry from the plan. When the plan is applied, the
  original column is restored.
- `0` or `+0` gives `scale = 1.0` and leaves the model unchanged.

Column scaling is deliberately simple: it changes the strength of the neuron's
output without touching the up or gate projections, and it is reversible.

## Target projection

`--target` selects which MLP projection is edited:

| Target | What is scaled | Shape |
|--------|----------------|-------|
| `down_proj` (default) | Column `i` | `(hidden, intermediate)` |
| `up_proj` | Row `i` | `(intermediate, hidden)` |
| `gate_proj` | Row `i` | `(intermediate, hidden)` |
| `all` | Row `i` of up and gate, column `i` of down | all three |

- `down_proj` changes the neuron's output contribution.
- `up_proj` changes the neuron's input projection.
- `gate_proj` changes the neuron's gating.
- `all` removes or amplifies the neuron at every stage of the MLP.

For example, this deletes neuron 8949 in layer 14 from all three projections:

```bash
neuron-tweak <model> --profile custom \
  --custom "L14 N8949 -1" --target all --save-directory ./out
```

## Interactive editor

```bash
neuron-tweak Qwen/Qwen3-4B-Instruct-2507 --profile custom
```

After the model loads, the editor shows the current plan and offers:

1. Add or update a neuron edit
2. Remove an entry from the plan
3. Clear the plan
4. Show syntax help
5. Done

When you choose Done, the tool asks for confirmation, applies the plan, prints
the changed column norms, and asks where to save the edited model. An empty
save directory keeps the edit in memory only.

## Non-interactive use

```bash
# One or more --custom arguments.
neuron-tweak Qwen/Qwen3-4B-Instruct-2507 --profile custom \
  --custom "L14 N8949 +1, L13 N7956 -1" \
  --save-directory ./qwen-custom

# Read a plan from a text file.
neuron-tweak Qwen/Qwen3-4B-Instruct-2507 --profile custom \
  --custom-file configs/custom.example.txt \
  --save-directory ./qwen-custom

# Parse and validate syntax without loading the model.
neuron-tweak --profile custom --custom "L14 N8949 +1" --dry-run
```

A plan file may contain one entry per line, commas, or semicolons. Empty lines
and lines starting with `#` are ignored.

## Applying and resetting

- Applying a new plan first restores every previously edited column, then
  applies the new plan from the original weights. Edits are absolute, not
  cumulative.
- `runtime.reset_edits()` restores all edited columns and returns the number of
  restored neurons. The interactive editor exposes this as part of the apply
  flow; the CLI applies a plan from original weights on every run.
- Saving writes a normal Hugging Face model directory with safetensors shards
  and, when available, the tokenizer.

## Validation and limits

- Layer numbers are checked against the model's layer count.
- Neuron indices are checked against the layer's MLP intermediate size.
- Invalid entries are reported together before anything is changed.
- Custom Tweak currently supports Llama/Qwen/Mistral-style models with
  `model.model.layers[*].mlp.down_proj`. OPT-style `dense_4h_to_h` is also
  recognized. GPT-2-style Conv1D MLPs are rejected with a clear error.
- Full-precision weights are required. Quantized (for example 4-bit) models must
  be loaded in float16 or bfloat16 for custom editing.
- Large positive scales can damage the model. Start with small changes, keep a
  copy of the original weights, and compare generations before and after.
