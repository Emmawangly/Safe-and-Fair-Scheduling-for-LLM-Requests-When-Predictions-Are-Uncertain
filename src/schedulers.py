def fcfs_scheduler(queue, **kwargs):
    if not queue:
        return None
    first_arrival_time = queue[0].arrival_time
    selected_index = 0
    for i in range(1, len(queue)):
        if queue[i].arrival_time < first_arrival_time:
            first_arrival_time = queue[i].arrival_time
            selected_index = i
    return selected_index

def ltr_scheduler(queue, **kwargs):
    if not queue:
        return None
    smallest_mu = queue[0].predicted_mu
    selected_index = 0
    for i in range(1, len(queue)):
        if queue[i].predicted_mu < smallest_mu:
            smallest_mu = queue[i].predicted_mu
            selected_index = i
    return selected_index

 

def robust_scheduler(queue, alpha, beta):
    if not queue:
        return None

    
    total_wait = sum(req.wait_time for req in queue)
    avg_wait = total_wait / len(queue) if queue else 0.0

    
    if avg_wait == 0.0:
        avg_wait = 1.0

    best_index = None
    best_score = float("-inf")

    for idx, req in enumerate(queue):
        mu = req.predicted_mu
        sigma = req.predicted_sigma
        T_wait = req.wait_time

        
        denom = max(mu + alpha * sigma, 1e-6)
        term1 = 1.0 / denom

        
        aging = beta * (T_wait / avg_wait)

        score = term1 + aging

        if score > best_score:
            best_score = score
            best_index = idx

    return best_index