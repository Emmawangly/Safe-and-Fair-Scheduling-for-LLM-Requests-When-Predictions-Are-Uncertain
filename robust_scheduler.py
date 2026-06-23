from typing import Optional, List
from shared_structures import RequestPacket

def robust_scheduler(queue: List[RequestPacket], alpha: float = 1.0, beta: float = 1.0, **kwargs) -> Optional[int]:
    if not queue:
        return None

    total_wait = sum(req.wait_time for req in queue)
    avg_wait = total_wait / len(queue)

    if avg_wait == 0.0:
        avg_wait = 1.0

    best_index = 0
    best_score = float("-inf")

    for idx, req in enumerate(queue):
        denom = max(req.predicted_mu + alpha * req.predicted_sigma, 1e-6)
        score = 1.0 / denom + beta * (req.wait_time / avg_wait)

        if score > best_score:
            best_score = score
            best_index = idx

    return best_index


if __name__ == "__main__":
    test_queue = [
        RequestPacket(
            request_id=1,
            arrival_time=0.0,
            actual_blocks=12,
            predicted_mu=10.0,
            predicted_sigma=2.0,
            wait_time=3.0
        ),
        RequestPacket(
            request_id=2,
            arrival_time=1.0,
            actual_blocks=7,
            predicted_mu=6.0,
            predicted_sigma=1.0,
            wait_time=5.0
        ),
        RequestPacket(
            request_id=3,
            arrival_time=2.0,
            actual_blocks=20,
            predicted_mu=18.0,
            predicted_sigma=5.0,
            wait_time=1.0
        ),
    ]

    result = robust_scheduler(test_queue, alpha=1.0, beta=1.0)
    expected = 1
    assert result == expected, f"Expected index {expected}, got {result}"
    print(f"Test passed: found index {result} (ID: {test_queue[result].request_id})")

    assert robust_scheduler([]) is None
    print("Test passed: empty queue returns None")

    zero_wait_queue = [
        RequestPacket(1, 0.0, 10, 10.0, 1.0, 0.0),
        RequestPacket(2, 0.0, 5,  3.0,  0.5, 0.0),
    ]
    assert robust_scheduler(zero_wait_queue, alpha=1.0, beta=1.0) == 1
    print("Test passed: zero wait time works")

    edge_queue = [RequestPacket(1, 0.0, 5, 0.0, 0.0, 1.0)]
    assert robust_scheduler(edge_queue, alpha=1.0, beta=1.0) == 0
    print("Test passed: no math errors with zero values")

    assert robust_scheduler(test_queue, alpha=0.5, beta=2.0) is not None
    print("Test passed: extra arguments work")

    print("\nAll tests finished successfully.")