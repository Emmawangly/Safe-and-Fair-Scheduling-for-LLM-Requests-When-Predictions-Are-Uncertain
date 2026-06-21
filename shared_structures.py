request_packet = {
    "request_id": int,         
    "arrival_time": float,     # Virtual time in seconds
    "actual_blocks": int,      # real count of KV-cache block 
    "predicted_mu": float,     # Predicted block count with error 
    "predicted_sigma": float,  # Uncertainty of the prediction - standard deviation 
    "wait_time": float,        # Seconds waited in the queue
}