def fcfs_scheduler(queue):
    if not queue:
        return None
    return 0 

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

        
        denom = mu + alpha * sigma
        

        
        term1 = 1.0 / denom

        
        aging = beta * (T_wait / avg_wait)

        score = term1 + aging

        if score > best_score:
            best_score = score
            best_index = idx

    return best_index