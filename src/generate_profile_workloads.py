"""
Generate reproducible profile-based workload files.

This script loads a block-length profile, generates workloads for
multiple prediction-error levels, and confirms that actual workloads
remain identical across error settings.
"""

import json
from pathlib import Path

from lmsys_profile import load_profile
from workload_generator import generate_workload


ERROR_RATES = [0, 20, 40, 60, 80]
NUM_REQUESTS = 1000
SEED = 42
DATA_DIR = Path("data")


def save_workload(queue, output_path):
    """Save one generated workload as formatted JSON."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(queue, file, indent=2)


def get_actual_blocks(queue):
    """Return the true block requirements for comparison."""
    return [request["actual_blocks"] for request in queue]


def main():
    profile_path = DATA_DIR / "lmsys_block_profile.json"

    profile = load_profile(str(profile_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    reference_actual_blocks = None

    for error_rate in ERROR_RATES:
        queue = generate_workload(
            num_requests=NUM_REQUESTS,
            error_rate_percent=error_rate,
            seed=SEED,
            block_profile=profile,
        )

        current_actual_blocks = get_actual_blocks(queue)

        if reference_actual_blocks is None:
            reference_actual_blocks = current_actual_blocks
        elif current_actual_blocks != reference_actual_blocks:
            raise RuntimeError(
                "Actual blocks changed across error rates. "
                "Only prediction error should change."
            )

        output_path = DATA_DIR / f"workload_lmsys_error_{error_rate}.json"
        save_workload(queue, output_path)

        print(
            f"Saved {len(queue)} requests with {error_rate}% prediction "
            f"error to {output_path}"
        )

    print()
    print("Validation passed:")
    print("All five workloads use identical actual_blocks values.")
    print("Only predicted_mu and predicted_sigma change by error rate.")


if __name__ == "__main__":
    main()