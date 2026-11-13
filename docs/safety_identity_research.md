# Safety and identity research notes

This document records the research background for the two workflows. It is a
living note, not a claim that any specific result will transfer to every model.

## Refusal behavior

Refusal in chat models is not a single mechanism, but several studies find a
low-dimensional representation that strongly predicts refusal. Arditi et al.
(2024) show that ablating one residual direction removes refusals across
several open models while preserving general capability on standard benchmarks.
Heretic builds on this with arbitrary-rank ablation and hyperparameter search.

Neuron Tweak adds a finer-grained view. Safety tuning tends to create sparse,
high-activation MLP neurons on refusal-inducing prompts. Suppressing the
neurons with the largest positive separation between refusal and harmless
prompts can reduce refusals with less collateral change than a full residual
ablation. The two approaches are complementary, so the safety modifier keeps an
optional hybrid residual ablation.

## Identity behavior

Models answer identity questions, such as their name, developer, owner, and
founder, using knowledge stored in their weights. Work on factual recall in
transformers localizes much of this knowledge in MLP blocks, where individual
neurons act as key-value features. Identity facts are a special case: they are
learned during pretraining and instruction tuning, and they may also be
reinforced by system prompts and chat templates.

The identity workflow therefore does two things:

1. Localize neurons whose activation predicts the current identity tokens,
   using logit-lens projection and causal patching.
2. Optimize a small, reversible weight edit on those neurons so that the target
   identity becomes the model's default answer, while unrelated factual prompts
   stay close to the original model.

Because the chat template and system prompt also influence identity answers,
the tool measures identity match with the model's normal chat template. The
edit is weights-only in v1; template changes are left to the user.

## Evaluation cautions

- Refusal rates measured with keyword lists can be noisy. Use a classifier
  scorer as a second opinion when possible.
- Identity edits may generalize unevenly across paraphrases. The 600-prompt
  set includes many phrasings and a held-out evaluation split for this reason.
- Always report KL divergence and a capability benchmark alongside the target
  metric. A lower refusal rate is not useful if the model becomes incoherent.
