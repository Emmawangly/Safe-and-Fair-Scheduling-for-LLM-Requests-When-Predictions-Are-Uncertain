"""
Extract reusable request-length profiles from LMSYS-style datasets.

Supported input formats:
- JSONL: useful for small local samples and testing
- Parquet: useful for larger real datasets

The output is a JSON block profile that can be passed to:
generate_workload(..., block_profile=profile)
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from lmsys_profile import build_block_profile, save_profile


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text length.

    Temporary approximation:
    about 4 characters per token.

    This can later be replaced with a model-specific tokenizer.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned_text = text.strip()

    if not cleaned_text:
        return 0

    return math.ceil(len(cleaned_text) / 4)


def normalize_conversation(value: Any) -> List[Dict[str, object]]:
    """
    Convert a conversation value into a list of message dictionaries.

    Supports:
    - already-parsed Python lists
    - JSON strings containing a list of messages
    """
    if isinstance(value, list):
        return [
            message
            for message in value
            if isinstance(message, dict)
        ]

    if isinstance(value, str):
        stripped_value = value.strip()

        if not stripped_value:
            return []

        try:
            parsed_value = json.loads(stripped_value)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed_value, list):
            return [
                message
                for message in parsed_value
                if isinstance(message, dict)
            ]

    return []


def extract_message_texts(record: Dict[str, object]) -> List[str]:
    """
    Extract non-empty message text from a conversation record.

    The default expected field is 'conversation'.
    A fallback 'messages' field is also supported.
    """
    conversation_value = record.get(
        "conversation",
        record.get("messages", []),
    )

    conversation = normalize_conversation(conversation_value)

    texts: List[str] = []

    for message in conversation:
        content = message.get("content", "")

        if isinstance(content, str) and content.strip():
            texts.append(content)

    return texts


def record_to_total_tokens(record: Dict[str, object]) -> int:
    """
    Estimate total context length for one conversation record.

    We include all user and assistant messages because KV-cache grows
    with the text processed by the model.
    """
    return sum(
        estimate_tokens(text)
        for text in extract_message_texts(record)
    )


def read_jsonl_token_lengths(
    input_path: str,
    max_records: Optional[int] = None,
) -> List[int]:
    """
    Read JSONL data and return estimated total token lengths.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    token_lengths: List[int] = []
    processed_records = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if max_records is not None and processed_records >= max_records:
                break

            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError:
                continue

            if not isinstance(record, dict):
                continue

            total_tokens = record_to_total_tokens(record)

            if total_tokens > 0:
                token_lengths.append(total_tokens)

            processed_records += 1

    if not token_lengths:
        raise ValueError("No valid token lengths were extracted")

    return token_lengths


def read_parquet_token_lengths(
    input_path: str,
    max_records: Optional[int] = None,
) -> List[int]:
    """
    Read Parquet data and return estimated total token lengths.

    The Parquet file must contain either:
    - a 'conversation' column, or
    - a 'messages' column.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    dataframe = pd.read_parquet(path)

    conversation_column = None

    for candidate in ("conversation", "messages"):
        if candidate in dataframe.columns:
            conversation_column = candidate
            break

    if conversation_column is None:
        raise ValueError(
            "Parquet file must contain a 'conversation' or 'messages' column"
        )

    if max_records is not None:
        dataframe = dataframe.head(max_records)

    token_lengths: List[int] = []

    for conversation_value in dataframe[conversation_column]:
        record = {
            conversation_column: conversation_value,
        }

        total_tokens = record_to_total_tokens(record)

        if total_tokens > 0:
            token_lengths.append(total_tokens)

    if not token_lengths:
        raise ValueError("No valid token lengths were extracted")

    return token_lengths


def read_dataset_token_lengths(
    input_path: str,
    max_records: Optional[int] = None,
) -> List[int]:
    """
    Automatically select the reader based on file extension.
    """
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return read_jsonl_token_lengths(
            input_path=str(path),
            max_records=max_records,
        )

    if suffix == ".parquet":
        return read_parquet_token_lengths(
            input_path=str(path),
            max_records=max_records,
        )

    raise ValueError(
        "Unsupported input format. Use a .jsonl or .parquet file."
    )


def build_profile_from_dataset(
    input_path: str,
    output_path: str,
    max_records: Optional[int] = None,
    tokens_per_block: int = 16,
) -> Dict[str, object]:
    """
    Build and save a KV-cache block profile from JSONL or Parquet data.
    """
    token_lengths = read_dataset_token_lengths(
        input_path=input_path,
        max_records=max_records,
    )

    profile = build_block_profile(
        token_lengths=token_lengths,
        tokens_per_block=tokens_per_block,
    )

    profile["source"] = "LMSYS-style dataset"
    profile["input_file"] = str(input_path)
    profile["token_count_method"] = "character_estimate_4_chars_per_token"

    save_profile(profile, output_path)

    return profile


def build_profile_from_jsonl(
    input_path: str,
    output_path: str,
    max_records: Optional[int] = None,
    tokens_per_block: int = 16,
) -> Dict[str, object]:
    """
    Backward-compatible wrapper for JSONL input.
    """
    return build_profile_from_dataset(
        input_path=input_path,
        output_path=output_path,
        max_records=max_records,
        tokens_per_block=tokens_per_block,
    )


def main() -> None:
    """
    Demonstrate the local JSONL sample pipeline.
    """
    input_path = "data/lmsys_sample.jsonl"
    output_path = "data/lmsys_block_profile.json"

    profile = build_profile_from_dataset(
        input_path=input_path,
        output_path=output_path,
        max_records=1000,
    )

    print("Created LMSYS-style block profile:")
    print(f"  Sample count: {profile['sample_count']}")
    print(f"  Min blocks:   {profile['min_blocks']}")
    print(f"  Max blocks:   {profile['max_blocks']}")
    print(f"  Mean blocks:  {profile['mean_blocks']}")
    print(f"  Median:       {profile['median_blocks']}")
    print(f"  Saved to:     {output_path}")


if __name__ == "__main__":
    main()