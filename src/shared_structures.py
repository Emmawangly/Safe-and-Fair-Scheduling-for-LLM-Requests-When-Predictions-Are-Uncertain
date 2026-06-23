from dataclasses import dataclass

@dataclass
class RequestPacket:
    request_id: int
    arrival_time: float
    actual_blocks: int
    predicted_mu: float
    predicted_sigma: float
    wait_time: float = 0.0
    ttft: float = 0.0
    preemptions: int = 0