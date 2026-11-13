"""Verify the built-in prompt datasets.

Run from the repository root:

    python scripts/verify_prompt_datasets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXPECTED_COUNT = 600
PLACEHOLDERS = (
    "model_name",
    "vendor",
    "owner",
    "founder",
    "product_line",
    "assistant_persona",
)
PLACEHOLDER_PATTERN = re.compile(
    "|".join(r"\{" + placeholder + r"\}" for placeholder in PLACEHOLDERS)
)


def check_file(relative_path: str, require_placeholder: bool) -> list[str]:
    path = DATA / relative_path
    errors: list[str] = []

    if not path.is_file():
        return [f"{relative_path}: file not found"]

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if len(lines) != EXPECTED_COUNT:
        errors.append(
            f"{relative_path}: expected {EXPECTED_COUNT} prompts, found {len(lines)}"
        )

    duplicates = len(lines) - len(set(lines))
    if duplicates:
        errors.append(f"{relative_path}: {duplicates} duplicate prompts")

    non_ascii = [line for line in lines if any(ord(char) > 127 for char in line)]
    if non_ascii:
        errors.append(f"{relative_path}: {len(non_ascii)} non-ASCII lines")

    if require_placeholder:
        missing = [line for line in lines if not PLACEHOLDER_PATTERN.search(line)]
        if missing:
            errors.append(
                f"{relative_path}: {len(missing)} prompts without a placeholder"
            )

    status = "OK" if not errors else "FAIL"
    print(
        f"{status:<4} {relative_path:<32} "
        f"prompts={len(lines):>3} unique={len(set(lines)):>3}"
    )
    return errors


def main() -> int:
    checks = [
        ("refusal/good_prompts.txt", False),
        ("refusal/bad_prompts.txt", False),
        ("identity/good_prompts.txt", True),
        ("identity/bad_prompts.txt", False),
    ]

    errors: list[str] = []
    for relative_path, require_placeholder in checks:
        errors.extend(check_file(relative_path, require_placeholder))

    print()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("All prompt datasets passed verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
