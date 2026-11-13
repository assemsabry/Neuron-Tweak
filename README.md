# Neuron Tweak

Training-free neuron-level modification for local language models.

Neuron Tweak locates and modifies individual neurons inside a local model.
It supports three workflows:

- Safety Tweak: identify and suppress the neurons associated with safety
  refusals, while measuring the cost in general capability.
- Identity Tweak: identify and rewrite the neurons that encode the model's
  self-identity, such as its name, developer, owner, and founder.
- Custom Tweak: edit or delete specific MLP neurons by name, for example
  `L14 N8949 +1` and `L13 N7956 -1`, with full control over the final plan.

The tool is inspired by [Heretic](https://github.com/p-e-w/heretic) and follows
the same core methodology: contrastive prompt sets, activation analysis,
hyperparameter optimization, reversible LoRA-based edits, and an interactive
terminal workflow. All computation runs locally on the GPU.

Compared with Heretic, Neuron Tweak uses 600 prompts per set instead of 400,
and targets individual neurons rather than only residual directions.

## Status

Custom Tweak is implemented end to end: model loading, L/N/delta parsing,
add/update/remove editing, direct MLP `down_proj` column scaling, reset, and
model saving. Safety Tweak and Identity Tweak still require the
activation-analysis and optimization pipeline. The repository also contains
the startup banner, the workflow selection screen, the prompt loader, and the
curated prompt datasets.

## Installation

```bash
pip install -e .
```

A CUDA build of PyTorch is recommended. The tool loads the model on the GPU
when CUDA, ROCm, or MPS is available and reports the detected backend at
startup.

## Usage

```bash
neuron-tweak
neuron-tweak Qwen/Qwen3-4B-Instruct-2507
neuron-tweak --profile identity
```

Custom Tweak can be used interactively or from the command line:

```bash
# Open the interactive plan editor after loading the model.
neuron-tweak Qwen/Qwen3-4B-Instruct-2507 --profile custom

# Apply a plan directly and save the edited model.
neuron-tweak Qwen/Qwen3-4B-Instruct-2507 --profile custom \
  --custom "L14 N8949 +1, L13 N7956 -1" --save-directory ./qwen-custom

# Read a plan from a file and validate it without loading the model.
neuron-tweak --profile custom --custom-file edits.txt --dry-run
```

At startup the tool prints the Neuron Tweak banner and asks which neurons you
want to modify. The banner color defaults to green and can be changed with the
`NEURON_TWEAK_BANNER_COLOR` environment variable (any Rich color name).

## Custom Tweak syntax

```
L<layer> N<neuron> <delta>
```

- `L` is 1-based: `L14` is the 14th transformer block.
- `N` is a 0-based index into the MLP intermediate dimension.
- The scale applied to the neuron's `down_proj` column is `1 + delta`,
  clamped to `[0, max_scale]`.
- `+1` doubles the neuron's contribution.
- `-1` sets it to zero (deletes the neuron). `delete` is an alias for `-1`.
- `-0.5` halves it, and `+0.5` increases it by 50 percent.
- `reset` removes the entry from the plan and restores the original weights
  when the plan is applied.
- Entries can be separated by commas, semicolons, or newlines.

Editing `down_proj` is enough to control the neuron: the column maps that MLP
neuron back to the hidden dimension, so scaling it scales the neuron's
contribution to the residual stream without touching the up or gate
projections. See [`docs/custom_tweak.md`](docs/custom_tweak.md) for the full
guide, including the interactive editor, validation rules, and limits.

## Repository layout

```
configs/                 TOML configuration examples
data/
  safety/                600 good + 600 bad prompts
  identity/              600 good + 600 bad prompts
docs/                    methodology and dataset documentation
plan/                    project plan
scripts/                 maintenance and verification scripts
src/neuron_tweak/        package source
tests/                   automated tests
```

## Prompt datasets

| File | Lines | Purpose |
|------|-------|---------|
| `data/safety/good_prompts.txt` | 600 | Harmless requests that define desirable behavior. |
| `data/safety/bad_prompts.txt` | 600 | Refusal-triggering requests that define undesirable behavior. |
| `data/identity/good_prompts.txt` | 600 | Target-identity prompts with configurable placeholders. |
| `data/identity/bad_prompts.txt` | 600 | Current-identity prompts that define undesirable behavior. |

See `data/README.md` for the placeholder and split rules.

## Responsible use

Use Neuron Tweak only on models you own or are licensed to modify. The tool is
intended for research, local model customization, and white-label deployments.
Do not use modified models to deceive end users or to cause harm.

## License

AGPL-3.0-or-later.
