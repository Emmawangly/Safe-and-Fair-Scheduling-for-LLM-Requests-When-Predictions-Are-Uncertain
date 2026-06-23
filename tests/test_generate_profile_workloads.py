import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from lmsys_profile import load_profile
from workload_generator import generate_workload


def test_profile_based_workloads_keep_actual_blocks_identical():
    profile = load_profile(
        str(PROJECT_ROOT / "data" / "lmsys_block_profile.json")
    )

    error_rates = [0, 20, 40, 60, 80]
    workloads = {}

    for error_rate in error_rates:
        workloads[error_rate] = generate_workload(
            num_requests=100,
            error_rate_percent=error_rate,
            seed=42,
            block_profile=profile,
        )

    reference_actual_blocks = [
        request["actual_blocks"]
        for request in workloads[0]
    ]

    for error_rate in error_rates[1:]:
        current_actual_blocks = [
            request["actual_blocks"]
            for request in workloads[error_rate]
        ]

        assert current_actual_blocks == reference_actual_blocks


def test_profile_based_workloads_change_predictions_with_error_rate():
    profile = load_profile(
        str(PROJECT_ROOT / "data" / "lmsys_block_profile.json")
    )

    queue_0 = generate_workload(
        num_requests=100,
        error_rate_percent=0,
        seed=42,
        block_profile=profile,
    )

    queue_80 = generate_workload(
        num_requests=100,
        error_rate_percent=80,
        seed=42,
        block_profile=profile,
    )

    predicted_mu_0 = [
        request["predicted_mu"]
        for request in queue_0
    ]

    predicted_mu_80 = [
        request["predicted_mu"]
        for request in queue_80
    ]

    assert predicted_mu_0 != predicted_mu_80