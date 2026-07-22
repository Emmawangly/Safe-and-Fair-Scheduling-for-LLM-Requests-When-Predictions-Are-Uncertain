from typing import Optional, List
from shared_structures import RequestPacket


def ljf_scheduler(queue: List[RequestPacket], **kwargs) -> Optional[int]:
    if not queue:
        return None

    
    worst_idx = 0
    worst_mu = queue[0].predicted_mu

    for i in range(1, len(queue)):
        if queue[i].predicted_mu > worst_mu:
            worst_mu = queue[i].predicted_mu
            worst_idx = i

    return worst_idx


if __name__ == "__main__":
    from shared_structures import RequestPacket

    q = [
        RequestPacket(1, 0.0, 5, 3.0, 0.5),
        RequestPacket(2, 0.0, 10, 18.0, 1.0),
        RequestPacket(3, 0.0, 7, 6.0, 0.3),
    ]

    # quick test to check it still works
    result = ljf_scheduler(q)
    assert result == 1
    print(f"  ✓ LJF selected index {result} (request_id={q[result].request_id}, mu={q[result].predicted_mu})")

    assert ljf_scheduler([]) is None
    print("  ✓ empty queue returns None")

    ljf_scheduler(q, alpha=1.0, beta=1.0)
    print("  ✓ accepts kwargs")

    print("\nall tests passed.")
