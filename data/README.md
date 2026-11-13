# Prompt datasets

The datasets follow the contrastive structure used by Heretic, but with 600
prompts per set instead of 400. Each workflow has a "good" set and a "bad" set.

```
data/
  safety/
    good_prompts.txt    600 harmless requests (desirable behavior)
    bad_prompts.txt     600 refusal-triggering requests (undesirable behavior)
  identity/
    good_prompts.txt    600 target-identity prompts (desirable behavior)
    bad_prompts.txt     600 current-identity prompts (undesirable behavior)
```

Every file is UTF-8 plain text with one prompt per line. Empty lines and lines
starting with `#` are ignored by the loader. Each file contains exactly 600
unique prompts.

## Safety workflow

- `bad_prompts.txt`: requests that commonly trigger safety refusals. These are
  request-level test queries, not operational instructions. They mirror the
  structure of public refusal benchmarks and are used to locate refusal
  neurons and to measure refusal rate.
- `good_prompts.txt`: harmless, useful requests. They define the behavior that
  must be preserved and provide the contrast set for locating
  refusal-selective neurons.

## Identity workflow

- `bad_prompts.txt`: questions and paraphrases that elicit the model's current
  identity (its original name, developer, owner, founder, model family, and
  related facts). These define the behavior to be changed.
- `good_prompts.txt`: the same kinds of questions with the target identity
  supplied in the prompt. They define the desired behavior and provide the
  contrast set for locating identity neurons.

### Placeholders

Identity `good_prompts.txt` uses placeholders that the tool expands from the
`[target_identity]` configuration section:

| Placeholder | Meaning |
|-------------|---------|
| `{model_name}` | Target model or assistant name |
| `{vendor}` | Target developer or vendor |
| `{owner}` | Target owning organization |
| `{founder}` | Target founder |
| `{product_line}` | Target product line or model family |
| `{assistant_persona}` | Target assistant persona description |

Example configuration:

```toml
[target_identity]
model_name = "AcmeLM"
vendor = "Acme Corporation"
owner = "Acme Corporation"
founder = "Jane Doe"
product_line = "Acme Assist"
assistant_persona = "Acme Assist, the official Acme assistant"
```

## Train/evaluation split

The default configuration shuffles each set with a fixed seed, uses 500 prompts
for neuron localization and optimization, and holds out 100 for evaluation.
Set `split_seed` to null to keep file order instead. Evaluation prompts are
never used to select neurons.

## Responsible use

These datasets exist to analyze and modify models that the user owns or is
licensed to modify. Do not use them to build or deploy models intended to
cause harm or to deceive users about a model's origin.
