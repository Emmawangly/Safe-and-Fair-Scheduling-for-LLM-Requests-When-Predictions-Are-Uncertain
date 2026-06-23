#from mock_generator import generate_mock_requests
from workload_generator import generate_workload
from engine import SimulationEngine
from schedulers import robust_scheduler
from shared_structures import RequestPacket

def main():
    queue = generate_workload(
        num_requests=100,
        error_rate_percent=40,
        seed=42
        
    )
    requests = [
        RequestPacket(
            request_id=req["request_id"],
            arrival_time=req["arrival_time"],
            actual_blocks=req["actual_blocks"],
            predicted_mu=req["predicted_mu"],
            predicted_sigma=req["predicted_sigma"],
            wait_time=req["wait_time"],
        )
        for req in queue
    ]
    engine = SimulationEngine(requests, time_step=1.0, blocks_per_second=4)
    engine.run(max_time=200.0, alpha=1.0, beta=1.0, scheduler_fn=robust_scheduler)
    engine.write_csv("results_robust_40pct.csv")

if __name__ == "__main__":
    main()