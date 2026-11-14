# Neuron-Tweak Project Plan

Version: 0.1 (Draft)
Date: 2026-11-13
Status: Planning
License (planned): AGPL-3.0-or-later (same as Heretic)

## 1. Project Summary

Neuron-Tweak is a command-line tool for precise, training-free modification of
local large language models at the level of individual neurons. It is inspired
by Heretic (https://github.com/p-e-w/heretic) and reuses the same core
methodology: contrastive prompt sets, residual/activation analysis, a
TPE-based hyperparameter optimizer (Optuna), reversible LoRA-based weight
editing, scorer plugins, and an interactive terminal workflow.

The key difference from Heretic is the unit of intervention:

- Heretic works primarily on residual directions (abliteration, arbitrary-rank ablation).
- Neuron-Tweak works on individual neurons (MLP intermediate dimensions, and
  optionally attention head outputs) inside specific layers of a specific model,
  with named, purpose-built modification profiles.

Neuron-Tweak provides two built-in modification profiles:

1. Safety profile: locate and suppress the neurons associated with
   safety/refusal behavior, reducing refusals while preserving general capability.
2. Identity profile: locate and adjust the neurons that encode the model's
   self-identity (model name, developer/vendor, owner/founder, product line)
   and rewrite them to a user-specified target identity.

The tool runs entirely locally, through a terminal (cmd/PowerShell on Windows,
any POSIX shell on Linux/macOS), on models the user is legally entitled to
modify. No training, no fine-tuning, no external server required.

Hard requirements added during planning:

- The entire user interface is English-only and contains no emojis.
- All heavy matrix math (activation capture, neuron scoring, weight edits,
  evaluation batches) runs on the GPU through PyTorch. CPU is used only for
  small control logic and disk I/O.
- At startup the tool prints a blocky pixel-art banner spelling
  "NEURON TWEAK", visually similar to the SHADE banner shown during planning,
  but rendered in a non-blue color (default: green, configurable).
- After the banner, the user chooses exactly one workflow:
  "Safety Tweak", "Identity Tweak", or "Custom Tweak".
- The tool ships with 600 good and 600 bad prompts for the safety workflow,
  and 600 good and 600 bad prompts for the identity workflow (see
  Section 3.4).

### 1.1 Why this project

- Heretic-style direction ablation can be blunt: it removes a linear direction
  but does not tell you which parts of the network encode the behavior.
- Mechanistic interpretability research shows that factual knowledge
  (including "I am <model>, made by <company>") is strongly localized in MLP
  neurons. Safety tuning also produces identifiable, sparse sets of
  high-activation neurons.
- A neuron-level tool gives finer-grained control, better interpretability
  (you can inspect and report exactly which neurons were changed), and an
  natural foundation for white-label deployments of local models (identity)
  and refusal analysis (safety).

### 1.2 Non-goals (v1)

- No gradient-based training, RLHF, or weight pretraining of any kind.
- No graphical user interface. Terminal only.
- No remote/API-hosted inference in the modification pipeline (local files only).
- No support (initially) for editing retrieval-augmented or tool-use
  behavior. Focus is safety and identity.
- No claim of making any model safe; the tool edits behavior, it does not
  certify it.

### 1.3 Responsible use

Neuron-Tweak is intended for modifying models you own or are licensed to modify.
Removing safety behavior and rebranding identity are both dual-use. The project
documentation must state this clearly, and the default configuration must not
silently strip safety behavior from a model served to third parties without
explicit user action. Models processed with the tool should not be
misrepresented to end users in deceptive or harmful deployments.

## 2. Core Design Principles (inherited from Heretic)

1. Training-free. All modifications are closed-form or numerically optimized
   edits to weights or activations. No backpropagation through the model's
   training objective.
2. Contrast-driven. Behavior is isolated by contrasting two prompt sets
   (e.g., harmful vs. harmless; target-identity vs. current-identity).
3. Optimized, not hand-tuned. An Optuna TPE study co-optimizes modification
   parameters against configurable scorers (e.g., refusal rate down, KL
   divergence against the original model down).
4. Reversible by default. Modifications are applied as a LoRA adapter
   relative to the frozen base weights, so the original model can be restored
   at any time.
5. Measured. Every change is paired with a score. The user sees what the edit
   cost (capability/KL divergence) before choosing to save.
6. Local-first. Everything runs on the user's hardware, including 4-bit
   quantized loading for large models.

## 3. The Two Modification Profiles

### 3.1 Profile A: Safety neurons (refusal suppression)

Goal: find the small set of neurons most responsible for safety refusals and
suppress them, while minimizing damage to unrelated capabilities.

Contrastive data (configurable, defaults shown):

- Positive set ("trigger" prompts): mlabonne/harmful_behaviors (test[:100])
- Negative set ("baseline" prompts): mlabonne/harmless_alpaca (test[:100])
- Optional local text files: one prompt per line.

Method sketch:

1. Forward both sets through the model with forward hooks capturing
   activations of every layer's MLP up-projection, gate-projection, and
   down-projection outputs, plus attention output projection outputs.
2. For each layer l and neuron i, compute a separation score, e.g.
   - mean activation on positive set minus mean activation on negative set,
     normalized by pooled standard deviation (Cohen's d), and/or
   - a signed logistic-probe margin (L1-logistic regression on CPU),
   - with optional per-prompt winsorization to tame massive activations
     (same motivation as Heretic's winsorization_quantile).
3. Rank neurons per layer; select top-k via the optimizer (k is a search
   parameter, not a constant).
4. Suppression operators (selectable in config):
   - downscale: multiply the incoming weights of selected neurons
     (rows of W_gate / W_up, and the matching column of W_down) by a factor s in (0, 1]
   - clamp: zero the activation contribution of the neuron at inference time
     by scaling its outgoing down-projection column to 0
   - bias-shift: add a negative bias to the neuron pre-activation (for activations
     with an additive bias term)
   - ablation hybrid (optional): combine neuron suppression with a small
     arbitrary-rank ablation on the residual stream; off by default
   - all operators are implemented as a LoRA adapter where possible, so reset
     is exact.

Success criteria:

- Refusal keyword rate on held-out harmful prompts drops below a configured floor.
- KL divergence on held-out harmless prompts stays below a configured ceiling.
- MMLU/GSM8K retention within a configurable tolerance (benchmark scorer).

### 3.2 Profile B: Identity neurons (self-concept editing)

Goal: make the model answer identity questions with a user-specified identity
(model name, vendor/developer, owner/founder, product name, assistant persona)
instead of its original one.

Configuration (example):

```toml
[target_identity]
model_name = "AcmeLM"
vendor = "Acme Corporation"
owner = "Acme Corporation"
founder = "Jane Doe"
product_line = "Acme Assist"
assistant_persona = "Acme Assist, the official Acme assistant"
```

Contrastive data (configurable, defaults provided):

- Current-identity prompts (bad set): the 600 questions in
  `data/identity/bad_prompts.txt`, covering names, vendors, owners, founders,
  and related facts.
- Target-identity prompts (good set): the 600 prompts in
  `data/identity/good_prompts.txt`. They use placeholders that are expanded
  from the `[target_identity]` configuration section. The score compares token
  likelihoods, not just strings.
- Anchor/anti-spoof prompts (harmless controls): generic factual questions
  used only to measure KL divergence.

Method sketch (two-stage):

Stage 1 - Localization:

1. Capture activations on identity prompts with forward hooks (same capture
   machinery as the safety profile).
2. Score per-neuron correlation with identity-token logits using the
   logit-lens / projected-lens on the unembedding matrix: for each identity
   token t (e.g., "Qwen", "Alibaba", "Meta", "Llama"), compute the projection
   of each neuron's activation vector onto the unembedding direction of t.
   Neurons whose activation strongly and specifically predicts the current
   identity tokens are "identity neurons".
3. Validate localization with causal tracing (activation patching) on a small
   prompt subset: patch identified neurons from a target-identity-consistent
   run into a current-identity-consistent run and measure the shift in
   identity-token logit; keep neurons with the highest causal effect.

Stage 2 - Editing:

1. Compute the desired direction as the difference of mean residual/neuron
   activations between current-identity and desired-completion contexts, or
   optimize a small Low-Rank edit on the selected neurons' weights so the
   target identity tokens are produced with high probability on the identity
   prompt set while random factual prompts are largely unaffected.
2. Apply as a LoRA adapter (rank configurable, default 8) so the edit is
   reversible. Optionally merge and export as a full model.
3. Optionally rewrite static identity strings in the model's `generation_config`,
   chat template, and tokenizer-adjacent metadata? No: weights only in v1.
   Config files describing the model identity (e.g., custom system prompt
   suggestions) are left to the user.

Success criteria:

- Identity match rate: fraction of held-out identity questions whose response
  contains the target name/vendor and does NOT contain the original ones,
  above a configured floor (default 0.8).
- KL divergence on harmless controls below a configured ceiling.
- No regression beyond tolerance on a small capability benchmark (PIQA default).

### 3.3 Relationship between the two profiles

The two profiles share the entire pipeline (loading, capture, scoring
infrastructure, optimization, export). Only the prompt sets, the per-neuron
scoring function, and the edit operator differ. A run selects one profile.
Running both is out of scope for v1; sequential runs are supported (safety
first, then identity, or vice versa) because each edit is an independent LoRA.

### 3.4 Built-in prompt datasets (600 good + 600 bad per workflow)

Heretic drives model analysis by sending large batches of prompts through the
model and scoring the responses. Its default configuration uses 400 desirable
("good") and 400 undesirable ("bad") prompts for the ARA modifier, plus
evaluation, KL-divergence, batch-size, and response-prefix prompt sets. Neuron
Tweak follows exactly the same principle, with a larger contrast set.

The initial release ships four curated prompt sets:

| File | Lines | Purpose |
|------|-------|---------|
| `data/safety/good_prompts.txt` | 600 | Harmless, useful requests that define the behavior to preserve. |
| `data/safety/bad_prompts.txt` | 600 | Refusal-triggering requests that define the behavior to remove. |
| `data/identity/good_prompts.txt` | 600 | Target-identity prompts with configurable placeholders. |
| `data/identity/bad_prompts.txt` | 600 | Current-identity prompts that define the behavior to change. |

Total: 2,400 prompts, 600 good and 600 bad for each of the two workflows.

Format: UTF-8 plain text, one prompt per line, no numbering; empty lines and
lines starting with `#` are ignored. The loader also accepts Hugging Face
datasets and local dataset files through the same `DatasetSpecification`
mechanism described in Section 5.

Identity "good" prompts use the placeholders `{model_name}`, `{vendor}`,
`{owner}`, `{founder}`, `{product_line}`, and `{assistant_persona}`. The tool
expands them from the `[target_identity]` configuration section at load time.
This keeps one static prompt file usable for any target identity.

The refusal bad prompts are request-level test queries, not operational
instructions. They are the functional equivalent of public refusal benchmarks
(for example `mlabonne/harmful_behaviors`) and exist to measure and localize
refusal behavior in models the user is authorized to modify. The identity bad
prompts cover names, vendors, owners, founders, model families, versions, and
many phrasings designed to prevent the identification from overfitting to one
question form.

By default each set is shuffled with a fixed seed: 500 prompts are used for
localization and optimization, and 100 are held out for evaluation. Set
`split_seed` to null to keep file order instead. The split is configurable.


## 4. System Architecture

```
neuron-tweak (console script)
  |
  +-- cli / main flow (terminal, questionary + rich)
  |     +-- argument parsing (--model, --config, --profile, flags)
  |     +-- environment/system probe (GPU, VRAM, dtype, batch size auto-tune)
  |     +-- response-prefix / CoT-skip detection
  |     +-- optimization study (Optuna TPE, journal storage, resumable)
  |     +-- trial selection menu (Pareto front)
  |     +-- actions: save / upload / chat / benchmark / reproduce
  |
  +-- config (TOML -> pydantic)
  |     +-- general settings (dtypes, quantization, batch size, max steps)
  |     +-- profile selection and profile-specific sections
  |     +-- scorer configuration (like Heretic's scorer plugins)
  |     +-- modifier configuration
  |
  +-- model runtime
  |     +-- transformers loading (dense, MoE, hybrid, multimodal as supported)
  |     +-- optional 4-bit bitsandbytes quantization
  |     +-- forward-hook activation capture with CPU offload
  |     +-- batching and response generation (greedy, fixed max length)
  |
  +-- analysis engine
  |     +-- capture engine: per-layer, per-neuron activation statistics
  |     +-- neuron scoring library (Cohens d, probe, causal patching)
  |     +-- layer/neuron selection with caching to disk (checkpoints/)
  |
  +-- modifiers (plugin system, mirrors Heretic's Modifier ABC)
  |     +-- safety_neurons.py  (Profile A)
  |     +-- identity_neurons.py (Profile B)
  |     +-- lora_editing.py     (shared LoRA apply/reset implementation)
  |
  +-- scorers (plugin system, mirrors Heretic's Scorer ABC)
  |     +-- keyword_rate.py     (refusal keywords / identity markers)
  |     +-- kl_divergence.py    (fidelity to original model)
  |     +-- identity_match.py   (target vs original identity string rate)
  |     +-- benchmark.py        (lm-eval-harness tasks, e.g. piqa, mmlu)
  |
  +-- export
        +-- merged model export (safetensors, max shard size)
        +-- LoRA adapter export (mergeable later)
        +-- Hugging Face upload with model card (optional)
        +-- reproducibility manifest (reproduce.json, Phase 6+)
```

### 4.1 Repository layout (planned)

```
neuron-tweak/
  pyproject.toml
  README.md
  LICENSE                       (AGPL-3.0)
  configs/
    default.toml                (shared defaults)
    safety.toml                 (safety workflow example)
    identity.toml               (identity workflow example)
  data/
    README.md                   (format, placeholders, responsible use)
    safety/
      good_prompts.txt          (600 harmless requests)
      bad_prompts.txt           (600 refusal-triggering requests)
    identity/
      good_prompts.txt          (600 target-identity prompts)
      bad_prompts.txt           (600 current-identity prompts)
  src/neuron_tweak/
    __init__.py
    __main__.py
    main.py                     (CLI entry, mirrors heretic/main.py)
    banner.py                   (blocky NEURON TWEAK banner, non-blue)
    prompts.py                  (prompt loading, placeholder expansion, split)
    neurons.py                  (L/N/delta spec parsing and edit plans)
    custom.py                   (Custom Tweak editor and apply pipeline)
    safety.py                   (Safety Tweak localization + suppression)
    config.py                   (pydantic settings from TOML)
    model.py                    (loading, MLP projection editing, saving)
    analysis.py                 (activation capture + Cohen's d scoring)
    optimizer.py                (Optuna study helpers, planned)
    evaluator.py                (scorer orchestration, planned)
    plugin.py                   (plugin ABC + loader)
    modifier.py                 (Modifier ABC)
    scorer.py                   (Scorer ABC)
    system.py                   (accelerator probe; batch-size auto-tune planned)
    utils.py
    modifiers/
      safety_neurons.py
      identity_neurons.py
      lora_editing.py
    scorers/
      keyword_rate.py
      identity_match.py
      kl_divergence.py
      benchmark.py
  tests/
    test_banner.py
    test_prompt_datasets.py
    test_prompts.py
    test_neurons.py
  scripts/
    verify_prompt_datasets.py
  docs/
    methodology.md
    prompt_datasets.md
    config_reference.md
    safety_identity_research.md
  plan/
    PLAN.md                     (this document)
```

### 4.2 CLI sketch

```
pip install neuron-tweak
neuron-tweak <model-name-or-path> [--profile safety|identity|custom] [--config config.toml]
neuron-tweak <model-name-or-path> --profile custom --custom "L14 N8949 +1, L13 N7956 -1"
neuron-tweak <model-name-or-path> --profile custom --custom-file edits.txt -o ./out
```

Interactive flow (mirrors Heretic):

0. Startup banner: blocky "NEURON TWEAK" pixel art, rendered in a non-blue
   color (default green, overridable via NEURON_TWEAK_BANNER_COLOR).
1. Hardware probe (CUDA/MPS/ROCm) and optimal batch size detection on GPU.
2. CoT/response-prefix detection.
3. Workflow selection: "Safety Tweak", "Identity Tweak", or "Custom Tweak".
4. Safety/Identity: neuron localization report (top neurons per layer, with
   scores), printed as a table; user may pin layers or exclude layers.
5. Safety/Identity: Optuna study runs trials; each trial proposes (top_k,
   per-layer scales, lora rank, edit operator) and prints scores.
6. Pareto-optimal trial selection menu.
7. Action menu: save / upload / chat / benchmark / reproduce.

Custom Tweak bypasses steps 4-6: after model loading it opens an editor where
the user adds, updates, or removes L/N/delta entries, applies them to the
selected MLP projection (`down_proj` by default, or `up_proj`, `gate_proj`, or
`all`), and saves the edited model.

Non-interactive flags (for scripting, mirroring Heretic's settings):

- --profile, --n-trials, --trial-index, --save-directory, --model-action,
  --evaluate-model, --checkpoint-action, --export-strategy
- Custom Tweak: --custom, --custom-file, --max-scale, --target, --interactive,
  --dry-run, --dtype, --device-map, --quantization, --trust-remote-code

## 5. Configuration Philosophy

- One active config file: `config.toml` in the working directory, or the file
  in the `NEURON_TWEAK_CONFIG_FILE` environment variable.
- TOML sections: `[target_identity]`, `[modifier.SafetyNeurons]`,
  `[modifier.IdentityNeurons]`, `[scorer.KeywordRate]`, etc. Same plugin-config
  convention as Heretic (`[scorer.<ClassName>]` with optional instance
  overrides).
- Sensible defaults: running `neuron-tweak <model>` with no config performs
  the safety profile with default datasets.
- Custom Tweak is configured from the CLI or its editor; it does not use a
  prompt dataset or an Optuna study.
- All datasets can be: Hugging Face dataset IDs (pinned to a commit), local
  paths, or plain text files with one prompt per line.

## 6. Optimization Objective

Default objectives (co-optimized, Pareto):

- Safety profile: minimize RefusalRate (keyword or classifier), minimize KL divergence.
- Identity profile: maximize IdentityMatch (or minimize IdentityMismatch),
  minimize KL divergence.
- Custom profile: no objective. The user supplies exact L/N/delta edits, picks
  the target projection, and decides when the plan is complete.
- Optional: minimize BenchmarkPenalty (e.g., PIQA accuracy drop).

The optimizer's search space (per profile):

| Parameter            | Range            | Notes                          |
|----------------------|------------------|--------------------------------|
| top_k per layer      | 4 .. 512         | log scale                      |
| layer selection      | band around      | default band = middle third of |
|                      | configurable band| layers (identity)              |
| suppression strength s | 0.0 .. 1.0      | multiplicative                 |
| lora_rank            | 1 .. 32          | for edit operators             |
| edit operator        | downscale/clamp/bias | categorical               |
| winsorization q      | 0.9 .. 1.0       | outlier taming                 |

## 7. Model Support Plan

Tier 1 (v1 target):

- Dense decoder-only Transformers with standard MLP blocks
  (Llama family, Qwen2/Qwen3, Mistral, Gemma 2/3, Phi-3/4).
- Behavior verified on Qwen3-4B-Instruct-2507 as reference model.

Tier 2 (best effort):

- MoE models (Mixtral, Qwen-MoE, gpt-oss): expert neurons handled by applying
  capture/edit per expert.
- Hybrid models (Mamba/attention): neuron editing only on MLP/GLU blocks.
- Multimodal models: text tower only.

Tier 3 (out of scope for v1):

- Encoder-only models, SSM-only models, models with non-standard activations
  lacking documented hooks.

## 8. Development Phases and Milestones

Phase 0 - Scaffolding (target: 1 week)

- Repository layout, pyproject.toml, uv setup, license, README skeleton.
- CLI skeleton with --help, --version.
- Config system with TOML loading and validation.
- CI: lint (ruff), type check (ty or mypy), pytest, coverage.
- Milestone: `neuron-tweak --help` works.

Phase 1 - Model runtime (target: 1-2 weeks)

- Model loading (transformers + accelerate), dtypes fallback, bnb_4bit option.
- Batch size auto-detection (Heretic-style tokens/s benchmark).
- Response generation with fixed greedy decoding, max response length.
- CoT block / response-prefix detection.
- Forward-hook activation capture with optional CPU offload.
- Milestone: capture activations for one layer of one model and dump to disk.

Phase 2 - Analysis engine (target: 1-2 weeks)

- Neuron scoring library (Cohen's d, logistic probe on CPU, logit-lens
  projection).
- Disk-cached capture results under checkpoints/.
- Localization report rendering (rich tables).
- Unit tests on synthetic tiny models (random weights) with known planted
  "neurons" verifying detection accuracy.
- Milestone: given a tiny synthetic model, the engine recovers the planted
  neurons.

Phase 3 - Safety profile (target: 2 weeks)

- safety_neurons.py modifier (downscale, clamp, bias-shift).
- KeywordRate scorer (refusal keywords, ported semantics) + KLDivergence scorer.
- Default config.safety.toml.
- Validation runs on Qwen3-4B-Instruct-2507; document refusal-rate vs KL curve.
- Milestone: safety refusal rate reduced with KL below default ceiling on the
  reference model, measured on held-out prompts.

Phase 4 - Identity profile (target: 2 weeks)

- identity_neurons.py modifier (localization + LoRA editing).
- identity_match.py scorer (exact-match on target vs original markers,
  configurable marker lists).
- Default config.identity.toml with target_identity section.
- Validation: rewrite a small reference model's identity to a test label;
  verify match rate >= 0.8 and KL below ceiling.
- Milestone: identity rewrite demo recorded (terminal session log) for docs.

Phase 5 - Optimizer and study management (target: 1 week)

- Optuna TPE study with journal storage, resumable checkpoints.
- Pareto front selection menu (questionary + rich).
- Ctrl+C handling, graceful stop, settings persistence.
- Milestone: full optimization loop on reference model end to end.

Phase 6 - Export and actions (target: 1 week)

- Save merged model / LoRA adapter only; HF upload with model card; chat
  action; benchmark action (lm-eval harness).
- Reproducibility manifest (reproduce.json) follow-up.
- Milestone: export a modified model and reload it for inference.

Phase 7 - Hardening and release (target: 1-2 weeks)

- Test matrix across at least 3 model families.
- Documentation (methodology, config reference, FAQ, responsible-use).
- Packaging (pip/uv publish to PyPI as neuron-tweak).
- v0.1.0 release.

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Neuron-level edits less robust than direction ablation | Incomplete refusal removal or identity drift | Keep hybrid abliteration as optional modifier; co-optimize; KL ceiling |
| Identity edit generalizes poorly beyond memorized phrasings | Model reverts on reworded questions | Causal-patching-validated neuron selection; paraphrase-augmented prompt set; match rate measured on held-out phrasings |
| Massive activations distort neuron scores | Wrong neurons selected | Winsorization (default on); robust z-scores |
| Quantized (4-bit) edit quality degrades | Exported model differs from measured | Merge into bf16 copy; warn on 4-bit merge; document accuracy notes |
| MoE complexity | Broken assumptions on expert routing | Restrict v1 capture/edit to router-chosen experts per token batch; document |
| Overclaiming capability retention | User surprised by capability loss | Always print benchmark + KL before save; default thresholds conservative |
| Legal/ethical misuse | Reputational/legal harm | Responsible-use docs; clear licensing; no default deceptive defaults |

## 10. Evaluation Plan

Reference evaluations (run for every release):

1. Safety profile on Qwen3-4B-Instruct-2507:
   - Refusal rate (classifier + keyword) on 100 held-out harmful prompts.
   - KL divergence on 100 held-out harmless prompts.
   - MMLU (5-shot, 200 questions) accuracy delta.
2. Identity profile on the same model, target identity = fictional "AcmeLM":
   - Identity match rate on 50 held-out identity questions.
   - KL divergence on 100 harmless prompts.
   - MMLU delta.
3. Regression tests in CI: synthetic tiny model with planted neurons; assert
   localization recall >= 0.9 on planted neurons.

## 11. Technology Stack

- Python 3.10+
- PyTorch >= 2.6
- transformers, accelerate, safetensors, peft (LoRA), bitsandbytes (optional, 4-bit)
- Optuna (TPE + journal storage)
- pydantic + tomllib (config)
- questionary + rich (terminal UI)
- lm-eval (optional benchmarks, extra dependency)
- datasets (Hugging Face datasets or local text files)
- pytest, ruff, ty/mypy (development)
- Packaging: pyproject.toml + uv, console entry point `neuron-tweak`

## 12. Licensing and Attribution

- New code: AGPL-3.0-or-later (chosen to remain compatible with Heretic's
  license for the architectural patterns it references; see below).
- Heretic is AGPL-3.0; no Heretic code will be copied into Neuron-Tweak.
  Architecture and methodology references (config conventions, plugin ABCs,
  terminal flow) will be reimplemented with attribution in docs/methodology.md.
- Attribution section in README credits Heretic, abliteration research
  (Arditi et al. 2024), and the interpretability work on MLP neuron knowledge
  storage that motivates the identity profile.

## 13. Open Questions

1. Package name availability on PyPI (`neuron-tweak`) - to check before Phase 7.
2. Whether to bundle the identity prompt set (multiple languages) in-repo or
   generate it at build time. Tentative: bundle in `data/`, CC-BY-style
   attribution.
3. Whether the identity editor should also patch the tokenizer chat template
   `{{ model_name }}`-style variables. Tentative: no, keep weights-only; document.
4. Multi-modifier runs (safety + identity in one study) - deliberately
   deferred; sequential runs in v1.
5. Windows-specific terminal behavior (questionary on cmd.exe vs Windows
   Terminal) - to be tested in Phase 5/7.

## 14. Immediate Next Steps

1. Confirm plan with stakeholders; adjust scope (profiles, model priorities).
2. Create the 600-good/600-bad prompt datasets for both workflows (done in
   `data/`).
3. Create repository skeleton (Phase 0), including the startup banner and
   the workflow-selection prompt (done).
4. Implement the Custom Tweak core: L/N/delta parsing, model loading, MLP
   projection editing, reset, and saving (done).
5. Implement activation capture, Cohen's d neuron scoring, and the first
   Safety Tweak localization/suppression pipeline (done).
6. Add automated before/after refusal-rate evaluation on the held-out safety
   prompts (done with `--evaluate`; general response scoring remains pending).
7. Decide the exact default datasets for the identity profile and finalize
   the target-identity config schema.
8. Run a feasibility spike: activation capture + logit-lens on one model to
   confirm identity neurons are detectable and sparse.
9. Add Optuna optimization and Pareto trial selection for Safety and
   Identity (Safety v1 now runs the Optuna study via `--study`; full Pareto
   selection remains pending).

## 15. Current Implementation Status (2026-11-13)

Completed:

- `data/safety/good_prompts.txt`: 600 unique harmless prompts.
- `data/safety/bad_prompts.txt`: 600 unique refusal-triggering prompts.
- `data/identity/good_prompts.txt`: 600 unique target-identity prompts with
  placeholders.
- `data/identity/bad_prompts.txt`: 600 unique current-identity prompts.
- `src/neuron_tweak/banner.py`: blocky "NEURON TWEAK" banner, default green.
- `src/neuron_tweak/neurons.py`: L/N/delta parsing, add/update/remove plans,
  scale clamping, and validation.
- `src/neuron_tweak/model.py`: local model loading with dtype fallback,
  accelerator device map, MLP up/gate/down projection editing, reset, and
  saving.
- `src/neuron_tweak/custom.py`: interactive Custom Tweak editor and
  non-interactive apply/save pipeline.
- `src/neuron_tweak/analysis.py`: down-projection activation capture and
  Cohen's d neuron scoring.
- `src/neuron_tweak/safety.py`: Safety Tweak localization, top-neuron
  selection, suppression plan, apply, and save pipeline.
- `src/neuron_tweak/evaluator.py`: greedy response generation, refusal-rate
  scoring, and good-prompt response metrics.
- `src/neuron_tweak/scorer.py`: scorer base class and refusal-rate scorer.
- `src/neuron_tweak/optimizer.py`: Optuna neuron study helpers and the
  Safety Optuna objective.
- `src/neuron_tweak/system.py`: CUDA/ROCm/MPS/CPU backend detection.
- `src/neuron_tweak/main.py`: CLI with Safety Tweak, Identity Tweak, and
  Custom Tweak, including `--custom`, `--custom-file`, `--target`,
  `--layers`, `--top-neurons`, `--analysis-prompts`, `--dry-run`, `--evaluate`,
  `--study`, and `--save-directory`.
- `src/neuron_tweak/prompts.py`: prompt loading, placeholder expansion, and
  train/evaluation split.
- `src/neuron_tweak/config.py`: TOML loading, nested value lookup, config path
  resolution, and CLI default merging.
- `configs/`, `docs/`, `scripts/`, and `tests/` folder structure (86 tests
  passing, including tiny Llama editing, activation-capture, generation,
  evaluation, capability-aware Optuna study, identity pipeline, and config
  integration tests).
- `pyproject.toml`, `README.md`, and `data/README.md`.

Total built-in prompts: 2,400 (600 good and 600 bad for each workflow).

Not yet implemented:

- Identity Tweak logit-lens and causal-tracing refinement beyond the
  contrastive activation pipeline.
- Full multi-objective Pareto studies; the current study uses a scalar
  refusal-plus-capability score.
- Capability benchmarks beyond the refusal rate and mean response length.
- Batch-size auto-tuning and general response generation.
- Chat, benchmark, and upload actions.
- Quantized-model editing (the current runtime requires full precision).
