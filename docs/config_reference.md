# Configuration reference

Neuron Tweak loads `configs/default.toml` plus a workflow configuration such as
`configs/refusal.toml` or `configs/identity.toml`. A local `config.toml` in the
working directory takes precedence, or use `--config path/to/file.toml`.

## Top-level

| Key | Type | Meaning |
|-----|------|---------|
| `profile` | `"refusal"` or `"identity"` | Workflow to run. |

## `[model]`

| Key | Default | Meaning |
|-----|---------|---------|
| `dtypes` | `["auto", "float16", "bfloat16", "float32"]` | Dtypes tried during loading. |
| `quantization` | `"none"` | `"none"` or `"bnb_4bit"`. |
| `device_map` | `"auto"` | Accelerate device placement. |
| `max_batch_size` | `128` | Upper bound for batch-size tuning. |
| `max_response_length` | `100` | Maximum tokens generated per response. |
| `offload_outputs_to_cpu` | `true` | Move captures to CPU early. |

## `[prompts]`

| Key | Default | Meaning |
|-----|---------|---------|
| `data_dir` | `"data"` | Root of the prompt datasets. |
| `evaluation_size` | `100` | Held-out prompts per good/bad set. |
| `split_seed` | `0` | Shuffle seed before splitting; null keeps file order. |

## `[optimization]`

| Key | Default | Meaning |
|-----|---------|---------|
| `n_trials` | `100` | Number of Optuna trials. |
| `n_startup_trials` | `30` | Random exploration trials. |
| `study_checkpoint_dir` | `"checkpoints"` | Resumable study storage. |

## `[export]`

| Key | Default | Meaning |
|-----|---------|---------|
| `max_shard_size` | `"5GB"` | Maximum size per safetensors shard. |
| `modified_model_type` | `"tweaked"` | Label used in menus and model cards. |

## `[target_identity]` (identity workflow)

| Key | Meaning |
|-----|---------|
| `model_name` | Target model or assistant name. |
| `vendor` | Target developer or vendor. |
| `owner` | Target owning organization. |
| `founder` | Target founder. |
| `product_line` | Target product line or model family. |
| `assistant_persona` | Target assistant persona description. |

## Plugin sections

Scorers are configured under `[scorer.<ClassName>]` and modifiers under
`[modifier.<ClassName>]`. This follows the Heretic convention and allows
multiple instances through `instance_name` in the future.
