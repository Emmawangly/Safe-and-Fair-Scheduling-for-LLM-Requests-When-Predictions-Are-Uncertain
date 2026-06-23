import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from lmsys_profile import (
    build_block_profile,
    load_profile,
    save_profile,
    tokens_to_blocks,
    validate_profile,
)


def test_tokens_to_blocks_uses_ceiling_division():
    assert tokens_to_blocks(1) == 1
    assert tokens_to_blocks(16) == 1
    assert tokens_to_blocks(17) == 2
    assert tokens_to_blocks(32) == 2
    assert tokens_to_blocks(33) == 3


def test_tokens_to_blocks_rejects_negative_values():
    with pytest.raises(ValueError):
        tokens_to_blocks(-1)


def test_tokens_to_blocks_rejects_invalid_block_size():
    with pytest.raises(ValueError):
        tokens_to_blocks(100, tokens_per_block=0)


def test_build_block_profile_has_correct_statistics():
    token_lengths = [8, 16, 17, 32, 33, 64, 128, 256]

    profile = build_block_profile(token_lengths)

    assert profile["tokens_per_block"] == 16
    assert profile["sample_count"] == 8
    assert profile["min_blocks"] == 1
    assert profile["max_blocks"] == 16
    assert profile["mean_blocks"] == 4.625
    assert profile["median_blocks"] == 2.5
    assert profile["block_lengths"] == [1, 1, 2, 2, 3, 4, 8, 16]


def test_build_block_profile_rejects_empty_input():
    with pytest.raises(ValueError):
        build_block_profile([])


def test_save_and_load_profile(tmp_path):
    profile = build_block_profile([16, 32, 48])

    output_path = tmp_path / "profile.json"

    save_profile(profile, str(output_path))
    loaded_profile = load_profile(str(output_path))

    assert loaded_profile == profile


def test_validate_profile_rejects_missing_fields():
    incomplete_profile = {
        "tokens_per_block": 16,
        "block_lengths": [1, 2, 3],
    }

    with pytest.raises(ValueError):
        validate_profile(incomplete_profile)


def test_validate_profile_rejects_invalid_block_lengths():
    invalid_profile = {
        "tokens_per_block": 16,
        "sample_count": 3,
        "min_blocks": 1,
        "max_blocks": 3,
        "mean_blocks": 2.0,
        "median_blocks": 2.0,
        "block_lengths": [1, 0, 3],
    }

    with pytest.raises(ValueError):
        validate_profile(invalid_profile)