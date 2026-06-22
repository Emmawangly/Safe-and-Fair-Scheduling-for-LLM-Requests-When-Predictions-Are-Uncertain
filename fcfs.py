from typing import Optional, List
from shared_structures import RequestPacket


def fcfs_scheduler(queue: List[RequestPacket]) -> Optional[int]:
    """
    First-Come-First-Served (FCFS) scheduler.

    Selects the index of the request with the earliest arrival_time.
    Ties are broken by queue position — the request that appears first
    in the list wins, preserving stable FCFS ordering.

    Note: FCFS does not use predicted_mu, predicted_sigma, or actual_blocks
    for ordering. However, it still suffers preemptions when the memory
    allocation (based on predicted_mu) is inaccurate — this is why FCFS
    degrades under high prediction error even though it ignores predictions.

    Args:
        queue: List of RequestPacket objects currently waiting in the queue.

    Returns:
        Index of the selected request in the queue, or None if the queue
        is empty.
    """
    if not queue:
        return None

    first_arrival_time = queue[0].arrival_time
    selected_index = 0

    for i in range(1, len(queue)):
        if queue[i].arrival_time < first_arrival_time:
            first_arrival_time = queue[i].arrival_time
            selected_index = i

    return selected_index


if __name__ == "__main__":
    test_queue = [
        RequestPacket(
            request_id=101,
            arrival_time=4.5,
            actual_blocks=12,
            predicted_mu=10.0,
            predicted_sigma=1.2,
            wait_time=0.0
        ),
        RequestPacket(
            request_id=102,
            arrival_time=1.2,
            actual_blocks=4,
            predicted_mu=5.0,
            predicted_sigma=0.5,
            wait_time=0.0
        ),
        RequestPacket(
            request_id=103,
            arrival_time=0.8,
            actual_blocks=32,
            predicted_mu=28.0,
            predicted_sigma=4.5,
            wait_time=0.0
        ),
        RequestPacket(
            request_id=104,
            arrival_time=7.1,
            actual_blocks=3,
            predicted_mu=3.0,
            predicted_sigma=0.1,
            wait_time=0.0
        ),
        RequestPacket(
            request_id=105,
            arrival_time=2.3,
            actual_blocks=8,
            predicted_mu=9.5,
            predicted_sigma=1.1,
            wait_time=0.0
        ),
    ]

    # ── Basic case: earliest arrival_time is request_id=103 at t=0.8 ────────
    result = fcfs_scheduler(test_queue)
    expected = 2
    assert result == expected, f"Expected index {expected}, got {result}"
    print(f"✅ Basic case passed — selected index {result} "
          f"→ request_id={test_queue[result].request_id}, "
          f"arrival_time={test_queue[result].arrival_time}")

    # ── Edge case: empty queue must return None ───────────────────────────────
    assert fcfs_scheduler([]) is None
    print("✅ Empty queue case passed — returned None")

    # ── Edge case: tie in arrival_time → first in queue wins ─────────────────
    tie_queue = [
        RequestPacket(request_id=1, arrival_time=1.0, actual_blocks=5,
                      predicted_mu=5.0, predicted_sigma=0.5, wait_time=0.0),
        RequestPacket(request_id=2, arrival_time=1.0, actual_blocks=3,
                      predicted_mu=3.0, predicted_sigma=0.2, wait_time=0.0),
    ]
    assert fcfs_scheduler(tie_queue) == 0
    print("✅ Tie-breaking case passed — first in queue wins")

    print("\nAll tests passed.")