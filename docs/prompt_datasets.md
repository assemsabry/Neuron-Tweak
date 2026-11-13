# Prompt datasets

The canonical documentation for the prompt datasets, including the good/bad
semantics, placeholder list, and train/evaluation split, lives in
[`data/README.md`](../data/README.md).

Summary:

| File | Lines | Role |
|------|-------|------|
| `data/safety/good_prompts.txt` | 600 | Harmless requests that define desirable behavior. |
| `data/safety/bad_prompts.txt` | 600 | Refusal-triggering requests that define undesirable behavior. |
| `data/identity/good_prompts.txt` | 600 | Target-identity prompts with placeholders. |
| `data/identity/bad_prompts.txt` | 600 | Current-identity prompts that define undesirable behavior. |

Custom Tweak does not use a prompt dataset. It edits the exact neurons supplied
by the user, so it has no good/bad contrast set.

Run `python scripts/verify_prompt_datasets.py` to check counts, uniqueness,
and placeholder coverage.
