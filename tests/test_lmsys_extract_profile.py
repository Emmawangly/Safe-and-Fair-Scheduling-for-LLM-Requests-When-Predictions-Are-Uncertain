import json
import sys
from pathlib import Path
import pandas as pd

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from lmsys_extract_profile import (
    estimate_tokens,
    extract_message_texts,
    read_dataset_token_lengths,
    read_jsonl_token_lengths,
    read_parquet_token_lengths,
    record_to_total_tokens,
)


def test_estimate_tokens_uses_four_characters_per_token():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_tokens_rejects_non_string_input():
    with pytest.raises(TypeError):
        estimate_tokens(123)


def test_extract_message_texts_reads_conversation_content():
    record = {
        "conversation": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    }

    assert extract_message_texts(record) == ["Hello", "Hi there"]


def test_extract_message_texts_skips_invalid_or_empty_messages():
    record = {
        "conversation": [
            {"role": "user", "content": "Valid message"},
            {"role": "assistant", "content": ""},
            {"role": "assistant"},
            "not a message dictionary",
        ]
    }

    assert extract_message_texts(record) == ["Valid message"]


def test_record_to_total_tokens_sums_all_messages():
    record = {
        "conversation": [
            {"role": "user", "content": "abcd"},
            {"role": "assistant", "content": "abcdefgh"},
        ]
    }

    assert record_to_total_tokens(record) == 3


def test_read_jsonl_token_lengths_reads_valid_records(tmp_path):
    input_path = tmp_path / "sample.jsonl"

    records = [
        {
            "conversation": [
                {"role": "user", "content": "abcd"},
                {"role": "assistant", "content": "abcdefgh"},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "abcdefghijkl"},
            ]
        },
    ]

    with input_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    token_lengths = read_jsonl_token_lengths(str(input_path))

    assert token_lengths == [3, 3]


def test_read_jsonl_token_lengths_skips_invalid_json_lines(tmp_path):
    input_path = tmp_path / "sample.jsonl"

    with input_path.open("w", encoding="utf-8") as file:
        file.write('{"conversation":[{"role":"user","content":"abcd"}]}\n')
        file.write("this is not valid json\n")
        file.write('{"conversation":[{"role":"assistant","content":"abcdefgh"}]}\n')

    token_lengths = read_jsonl_token_lengths(str(input_path))

    assert token_lengths == [1, 2]


def test_read_jsonl_token_lengths_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        read_jsonl_token_lengths("data/file_that_does_not_exist.jsonl")
def test_read_parquet_token_lengths_reads_conversation_column(tmp_path):
    input_path = tmp_path / "sample.parquet"

    dataframe = pd.DataFrame(
        {
            "conversation": [
                '[{"role":"user","content":"abcd"},'
                '{"role":"assistant","content":"abcdefgh"}]',
                '[{"role":"user","content":"abcdefghijkl"}]',
            ]
        }
    )

    dataframe.to_parquet(input_path)

    token_lengths = read_parquet_token_lengths(str(input_path))

    assert token_lengths == [3, 3]


def test_read_dataset_token_lengths_automatically_reads_parquet(tmp_path):
    input_path = tmp_path / "sample.parquet"

    dataframe = pd.DataFrame(
        {
            "messages": [
                '[{"role":"user","content":"abcd"}]',
                '[{"role":"assistant","content":"abcdefgh"}]',
            ]
        }
    )

    dataframe.to_parquet(input_path)

    token_lengths = read_dataset_token_lengths(str(input_path))

    assert token_lengths == [1, 2]