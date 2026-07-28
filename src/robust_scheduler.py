#This file is a record of the robust scheduler with the correct formula. It was included inside the schedulers.py, but we kept here as aregister of our scheduler

def robust_scheduler(queue, alpha, beta):
    if not queue:
        return None

    total_wait = sum(req.wait_time for req in queue)
    avg_wait = total_wait / len(queue)

    if avg_wait == 0.0:
        avg_wait = 1.0

    best_index = None
    best_score = float("-inf")

    for idx, req in enumerate(queue):
        
        
        mu = max(req.predicted_mu, 0.0)
        sigma = max(req.predicted_sigma, 0.0)
        wait_time = max(req.wait_time, 0.0)

        denom = max(mu + alpha * sigma, 1e-6)
        size_term = avg_wait / denom  # scale fixed here!

        aging_term = beta * (wait_time / avg_wait)

        score = size_term + aging_term

        if score > best_score:
            best_score = score
            best_index = idx

    return best_index


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from shared_structures import RequestPacket

    q = [
        RequestPacket(1, 0.0, 5, 3.0, 0.5, wait_time=2.0),
        RequestPacket(2, 0.0, 10, 18.0, 1.0, wait_time=0.0),
        RequestPacket(3, 0.0, 7, 6.0, 0.3, wait_time=1.0),
    ]

    result = robust_scheduler(q, alpha=0.5, beta=0.5)
    assert result is not None
    print(f"  ✓ robust picked index {result} (request_id={q[result].request_id})")

    assert robust_scheduler([], alpha=0.5, beta=0.5) is None
    print("  ✓ empty queue returns None")

    print("\nall tests passed ✓")
