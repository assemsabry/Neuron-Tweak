# Neuron Tweak

Training-free neuron-level modification for local language models.

Neuron Tweak locates and modifies individual neurons inside a local model.
It supports two workflows:

- Refusal neurons: identify and suppress the neurons associated with safety
  refusals, while measuring the cost in general capability.
- Identity neurons: identify and rewrite the neurons that encode the model's
  self-identity, such as its name, developer, owner, and founder.

The tool is inspired by [Heretic](https://github.com/p-e-w/heretic) and follows
the same core methodology: contrastive prompt sets, activation analysis,
hyperparameter optimization, reversible LoRA-based edits, and an interactive
terminal workflow. All computation runs locally on the GPU.

Compared with Heretic, Neuron Tweak uses 600 prompts per set instead of 400,
and targets individual neurons rather than only residual directions.

## Status

Early development. The current release contains the project skeleton, the
startup banner, the workflow selection screen, the prompt loader, and the
curated prompt datasets.

## Installation

```bash
pip install -e .
```

## Usage

```bash
neuron-tweak
neuron-tweak Qwen/Qwen3-4B-Instruct-2507
neuron-tweak --profile identity
```

At startup the tool prints the Neuron Tweak banner and asks which neurons you
want to modify. The banner color defaults to green and can be changed with the
`NEURON_TWEAK_BANNER_COLOR` environment variable (any Rich color name).

## Repository layout

```
configs/                 TOML configuration examples
data/
  refusal/               600 good + 600 bad prompts
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
| `data/refusal/good_prompts.txt` | 600 | Harmless requests that define desirable behavior. |
| `data/refusal/bad_prompts.txt` | 600 | Refusal-triggering requests that define undesirable behavior. |
| `data/identity/good_prompts.txt` | 600 | Target-identity prompts with configurable placeholders. |
| `data/identity/bad_prompts.txt` | 600 | Current-identity prompts that define undesirable behavior. |

See `data/README.md` for the placeholder and split rules.

## Responsible use

Use Neuron Tweak only on models you own or are licensed to modify. The tool is
intended for research, local model customization, and white-label deployments.
Do not use modified models to deceive end users or to cause harm.

## License

AGPL-3.0-or-later.
