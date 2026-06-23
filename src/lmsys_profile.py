"""
Utilities for building and using an LLM request-length profile.

This module converts token lengths into KV-cache block lengths,
summarizes a collection of block lengths, and samples realistic
block requirements from a saved profile.
"""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_TOKENS_PER_BLOCK = 16


def tokens_to_blocks(
    token_count: int,
    tokens_per_block: int = DEFAULT_TOKENS_PER_BLOCK,
) -> int:
    """
    Convert a token length into the number of KV-cache blocks required.

    Example:
        1 to 16 tokens  -> 1 block
        17 to 32 tokens -> 2 blocks
    """
    if token_count < 0:
        raise ValueError("token_count must be non-negative")

    if tokens_per_block <= 0:
        raise ValueError("tokens_per_block must be greater than zero")

    return max(1, math.ceil(token_count / tokens_per_block))


def build_block_profile(
    token_lengths: Iterable[int],
    tokens_per_block: int = DEFAULT_TOKENS_PER_BLOCK,
) -> Dict[str, object]:
    """
    Build a reusable KV-cache block profile from token lengths.

    The returned dictionary can later be saved as JSON and used by the
    workload generator to sample actual_blocks values.
    """
    block_lengths: List[int] = []

    for token_count in token_lengths:
        block_lengths.append(
            tokens_to_blocks(
                token_count=token_count,
                tokens_per_block=tokens_per_block,
            )
        )

    if not block_lengths:
        raise ValueError("token_lengths must contain at least one value")

    sorted_blocks = sorted(block_lengths)

    return {
        "tokens_per_block": tokens_per_block,
        "sample_count": len(sorted_blocks),
        "min_blocks": sorted_blocks[0],
        "max_blocks": sorted_blocks[-1],
        "mean_blocks": round(sum(sorted_blocks) / len(sorted_blocks), 3),
        "median_blocks": round(_median(sorted_blocks), 3),
        "block_lengths": sorted_blocks,
    }


def save_profile(profile: Dict[str, object], output_path: str) -> None:
    """
    Save a block-length profile to a JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)


def load_profile(profile_path: str) -> Dict[str, object]:
    """
    Load a previously saved block-length profile from JSON.
    """
    path = Path(profile_path)

    with path.open("r", encoding="utf-8") as file:
        profile = json.load(file)

    validate_profile(profile)

    return profile


def validate_profile(profile: Dict[str, object]) -> None:
    """
    Check that a loaded profile contains the fields needed by the generator.
    """
    required_fields = {
        "tokens_per_block",
        "sample_count",
        "min_blocks",
        "max_blocks",
        "mean_blocks",
        "median_blocks",
        "block_lengths",
    }

    missing_fields = required_fields - set(profile.keys())

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Profile is missing required fields: {missing}")

    block_lengths = profile["block_lengths"]

    if not isinstance(block_lengths, list) or not block_lengths:
        raise ValueError("profile['block_lengths'] must be a non-empty list")

    if any(
        not isinstance(block_count, int) or block_count < 1
        for block_count in block_lengths
    ):
        raise ValueError(
            "profile['block_lengths'] must contain positive integers"
        )


def _median(sorted_values: List[int]) -> float:
    """
    Return the median of a non-empty sorted list.
    """
    middle_index = len(sorted_values) // 2

    if len(sorted_values) % 2 == 1:
        return float(sorted_values[middle_index])

    return (
        sorted_values[middle_index - 1]
        + sorted_values[middle_index]
    ) / 2


def main() -> None:
    """
    Small local demonstration using sample token lengths.

    This does not use LMSYS data yet. It only verifies that the profile
    pipeline works before real LMSYS lengths are added.
    """
    sample_token_lengths = [8, 16, 17, 32, 33, 64, 128, 256]

    profile = build_block_profile(sample_token_lengths)

    output_path = "data/example_lmsys_block_profile.json"
    save_profile(profile, output_path)

    print("Created example block profile:")
    print(f"  Sample count: {profile['sample_count']}")
    print(f"  Min blocks:   {profile['min_blocks']}")
    print(f"  Max blocks:   {profile['max_blocks']}")
    print(f"  Mean blocks:  {profile['mean_blocks']}")
    print(f"  Median:       {profile['median_blocks']}")
    print(f"  Saved to:     {output_path}")


if __name__ == "__main__":
    main()