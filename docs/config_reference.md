# Configuration reference

Neuron Tweak loads `configs/default.toml` plus a workflow configuration such as
`configs/safety.toml` or `configs/identity.toml`. A local `config.toml` in the
working directory takes precedence, or use `--config path/to/file.toml`.

## Top-level

| Key | Type | Meaning |
|-----|------|---------|
| `profile` | `"safety"`, `"identity"`, or `"custom"` | Workflow to run. |

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

## Safety Tweak options (command line)

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
| `--seed` | `0` | Shuffle seed for the train/evaluation split. |
| `--yes` | off | Skip the apply confirmation. |
| `--dry-run` | off | Analyze and print, but do not change weights. |
| `--evaluate` | off | Generate before/after responses and report the refusal rate. |
| `--eval-max-new-tokens` | `64` | Max new tokens for evaluation responses. |
| `--eval-batch-size` | `4` | Batch size for evaluation responses. |
| `--study` | off | Run an Optuna study over the selected safety neurons. |
| `--study-trials` | `30` | Number of Optuna trials for the safety study. |
| `--study-startup-trials` | `10` | Startup trials for the Optuna safety study. |
| `--study-direction` | `minimize` | Study direction: `minimize` or `maximize`. |
| `--study-apply-best` | off | Apply the best safety study trial and save it. |
| `--study-capability-weight` | `0.5` | Weight of the good-prompt capability penalty in the study objective. |
| `--save-directory` | none | Directory for the edited model. |

The shared `--target`, `--max-scale`, `--save-directory`, `--dtype`,
`--device-map`, `--quantization`, and `--trust-remote-code` options also apply
to Safety Tweak.

## Custom Tweak options (command line)

Custom Tweak does not need a TOML section. Its plan is supplied through the
command line or the interactive editor.

| Option | Meaning |
|--------|---------|
| `--profile custom` | Select the Custom Tweak workflow. |
| `--custom "L14 N8949 +1"` | Add one or more edits; repeatable and comma-friendly. |
| `--custom-file PATH` | Read edits from a text file (`configs/custom.example.txt`). |
| `--max-scale N` | Maximum multiplicative scale (default `10`). |
| `--target PROJECTION` | `down_proj` (default), `up_proj`, `gate_proj`, or `all`. |
| `--interactive` | Open the editor even when `--custom` entries are present. |
| `--dry-run` | Parse and print the plan without loading the model. |
| `--save-directory PATH` | Save the edited model after applying the plan. |
| `--dtype`, `--device-map`, `--quantization` | Model loading controls. |

## Plugin sections

Scorers are configured under `[scorer.<ClassName>]` and modifiers under
`[modifier.<ClassName>]`. This follows the Heretic convention and allows
multiple instances through `instance_name` in the future.
