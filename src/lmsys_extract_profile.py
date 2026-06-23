"""
Extract a reusable request-length profile from LMSYS-style JSONL data.

This module does not download LMSYS data. It expects a local JSONL file
whose records contain a conversation field in OpenAI-style message format.

The output is a JSON block profile that can be passed to
generate_workload(..., block_profile=profile).
"""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from lmsys_profile import build_block_profile, save_profile


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text length.

    This is a simple temporary estimator: about 4 characters per token.
    Replace it with a model-specific tokenizer later if the group chooses one.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cleaned_text = text.strip()

    if not cleaned_text:
        return 0

    return math.ceil(len(cleaned_text) / 4)


def extract_message_texts(record: Dict[str, object]) -> List[str]:
    """
    Extract text content from an LMSYS-style conversation record.

    Expected conversation format:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    conversation = record.get("conversation", [])

    if not isinstance(conversation, list):
        return []

    texts: List[str] = []

    for message in conversation:
        if not isinstance(message, dict):
            continue

        content = message.get("content", "")

        if isinstance(content, str) and content.strip():
            texts.append(content)

    return texts


def record_to_total_tokens(record: Dict[str, object]) -> int:
    """
    Estimate total context length for one conversation record.

    We use all user and assistant messages because KV-cache grows with
    the context processed by the model.
    """
    message_texts = extract_message_texts(record)

    return sum(estimate_tokens(text) for text in message_texts)


def read_jsonl_token_lengths(
    input_path: str,
    max_records: Optional[int] = None,
) -> List[int]:
    """
    Read LMSYS-style JSONL data and return estimated total token lengths.

    Empty, invalid, or text-free records are skipped.
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


def build_profile_from_jsonl(
    input_path: str,
    output_path: str,
    max_records: Optional[int] = None,
    tokens_per_block: int = 16,
) -> Dict[str, object]:
    """
    Read local LMSYS-style JSONL data and save a KV-cache block profile.
    """
    token_lengths = read_jsonl_token_lengths(
        input_path=input_path,
        max_records=max_records,
    )

    profile = build_block_profile(
        token_lengths=token_lengths,
        tokens_per_block=tokens_per_block,
    )

    profile["source"] = "LMSYS-style JSONL"
    profile["token_count_method"] = "character_estimate_4_chars_per_token"
    profile["input_file"] = str(input_path)

    save_profile(profile, output_path)

    return profile


def main() -> None:
    """
    Demonstrate expected usage after a local LMSYS JSONL file is available.

    This command will only work after you put a JSONL file at the input path.
    """
    input_path = "data/lmsys_sample.jsonl"
    output_path = "data/lmsys_block_profile.json"

    profile = build_profile_from_jsonl(
        input_path=input_path,
        output_path=output_path,
        max_records=1000,
    )

    print("Created LMSYS-based block profile:")
    print(f"  Sample count: {profile['sample_count']}")
    print(f"  Min blocks:   {profile['min_blocks']}")
    print(f"  Max blocks:   {profile['max_blocks']}")
    print(f"  Mean blocks:  {profile['mean_blocks']}")
    print(f"  Median:       {profile['median_blocks']}")
    print(f"  Saved to:     {output_path}")


if __name__ == "__main__":
    main()