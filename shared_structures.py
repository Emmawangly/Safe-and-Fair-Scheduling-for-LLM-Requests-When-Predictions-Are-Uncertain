from dataclasses import dataclass

@dataclass
class RequestPacket:
    request_id: int         
    arrival_time: float        # Virtual time in seconds
    actual_blocks: int        # real count of KV-cache blocks
    predicted_mu: float      #  Predicted block count with error 
    predicted_sigma: float    # Uncertainty of the prediction 
    wait_time: float = 0.0    # Seconds waited in the queue
    ttft: float = 0.0          
    preemptions: int = 0