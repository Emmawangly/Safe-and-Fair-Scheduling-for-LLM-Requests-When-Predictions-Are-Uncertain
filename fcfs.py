from shared_structures import RequestPacket

def fcfs_scheduler(queue):
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
        )
    ]
    
    result_index = fcfs_scheduler(test_queue)
    print(f"Result index: {result_index}")