import random
from typing import Dict, Optional
import math
import json

def sample_actual_blocks(
    rng: random.Random,
    block_profile: Optional[Dict[str, object]] = None,
) -> int:
    """
    Return one actual KV-cache block requirement.

    Without a profile, keep the original temporary uniform distribution.
    With a profile, sample from the profile's block_lengths list.
    """
    if block_profile is None:
        return rng.randint(1, 20)

    block_lengths = block_profile["block_lengths"]

    if not isinstance(block_lengths, list) or not block_lengths:
        raise ValueError(
            "block_profile must contain a non-empty 'block_lengths' list"
        )

    return rng.choice(block_lengths)


def create_request(
    request_id,
    rng,
    arrival_time,
    error_rate_percent,
    block_profile=None,
):
    actual_blocks = sample_actual_blocks(rng, block_profile)
    error_fraction = error_rate_percent / 100
    noise_factor = rng.uniform(1 - error_fraction, 1 + error_fraction)
    predicted_mu = max(1.0, actual_blocks * noise_factor)
    predicted_sigma = actual_blocks * error_fraction / math.sqrt(3)

    request = {
        "request_id": request_id,
        "arrival_time": round(arrival_time, 3),
        "actual_blocks": actual_blocks,
        "predicted_mu": round(predicted_mu, 3),
        "predicted_sigma": round(predicted_sigma, 3),
        "wait_time": 0.0,
    }

    return request


def generate_workload(
    num_requests,
    error_rate_percent,
    seed=42,
    block_profile=None,
):
    rng = random.Random(seed)

    queue = []
    current_time = 0.0

    for request_id in range(1, num_requests + 1):
        interarrival_time = rng.expovariate(1.0)
        current_time += interarrival_time

        request = create_request(
    request_id,
    rng,
    current_time,
    error_rate_percent,
    block_profile,
)
        queue.append(request)

    return queue

def save_queue_as_json(queue, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(queue, file, indent=2)

def main():
    error_levels = [0, 20, 40, 60, 80]

    for error_rate_percent in error_levels:
        queue = generate_workload(
            num_requests=1000,
            error_rate_percent=error_rate_percent,
            seed=42,
        )

        output_path = f"data/workload_error_{error_rate_percent}.json"
        save_queue_as_json(queue, output_path)

        print(
            f"Saved {len(queue)} requests "
            f"with {error_rate_percent}% prediction error "
            f"to {output_path}"
        )

    sample_queue = generate_workload(
        num_requests=10,
        error_rate_percent=80,
        seed=42,
    )

    save_queue_as_json(sample_queue, "sample_output.json")

    print("\nSaved sample_output.json with 10 requests at 80% prediction error.")
    print("\nFirst 10 request packets:")

    for request in sample_queue:
        print(request)


if __name__ == "__main__":
    main()
