import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from workload_generator import generate_workload


REQUIRED_FIELDS = {
    "request_id",
    "arrival_time",
    "actual_blocks",
    "predicted_mu",
    "predicted_sigma",
    "wait_time",
}


def test_generate_workload_returns_correct_number_of_requests():
    queue = generate_workload(
        num_requests=100,
        error_rate_percent=40,
        seed=42,
    )

    assert len(queue) == 100


def test_each_request_has_required_fields():
    queue = generate_workload(
        num_requests=10,
        error_rate_percent=40,
        seed=42,
    )

    for request in queue:
        assert set(request.keys()) == REQUIRED_FIELDS


def test_request_field_types_are_correct():
    queue = generate_workload(
        num_requests=10,
        error_rate_percent=40,
        seed=42,
    )

    for request in queue:
        assert isinstance(request["request_id"], int)
        assert isinstance(request["arrival_time"], float)
        assert isinstance(request["actual_blocks"], int)
        assert isinstance(request["predicted_mu"], float)
        assert isinstance(request["predicted_sigma"], float)
        assert isinstance(request["wait_time"], float)


def test_arrival_times_are_non_decreasing():
    queue = generate_workload(
        num_requests=100,
        error_rate_percent=40,
        seed=42,
    )

    arrival_times = [request["arrival_time"] for request in queue]

    assert arrival_times == sorted(arrival_times)


def test_same_seed_generates_same_workload():
    queue_1 = generate_workload(
        num_requests=50,
        error_rate_percent=40,
        seed=42,
    )

    queue_2 = generate_workload(
        num_requests=50,
        error_rate_percent=40,
        seed=42,
    )

    assert queue_1 == queue_2


def test_zero_percent_error_has_perfect_prediction():
    queue = generate_workload(
        num_requests=50,
        error_rate_percent=0,
        seed=42,
    )

    for request in queue:
        assert request["predicted_mu"] == float(request["actual_blocks"])
        assert request["predicted_sigma"] == 0.0


def test_wait_time_starts_at_zero():
    queue = generate_workload(
        num_requests=50,
        error_rate_percent=80,
        seed=42,
    )

    for request in queue:
        assert request["wait_time"] == 0.0


def test_prediction_sigma_increases_with_error_rate():
    queue_0 = generate_workload(
        num_requests=10,
        error_rate_percent=0,
        seed=42,
    )

    queue_80 = generate_workload(
        num_requests=10,
        error_rate_percent=80,
        seed=42,
    )

    total_sigma_0 = sum(request["predicted_sigma"] for request in queue_0)
    total_sigma_80 = sum(request["predicted_sigma"] for request in queue_80)

    assert total_sigma_0 == 0.0
    assert total_sigma_80 > total_sigma_0

def test_generator_samples_actual_blocks_from_profile():
    block_profile = {
        "block_lengths": [3, 3, 3],
    }

    queue = generate_workload(
        num_requests=10,
        error_rate_percent=40,
        seed=42,
        block_profile=block_profile,
    )

    for request in queue:
        assert request["actual_blocks"] == 3