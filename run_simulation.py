from mock_generator import generate_mock_requests
from engine import SimulationEngine

def main():
    requests = generate_mock_requests()
    engine = SimulationEngine(requests, time_step=1.0, blocks_per_second=4)
    engine.run(max_time=50.0, alpha=1.0, beta=1.0)
    print("Completed:", len(engine.completed))
    for rec in engine.records:
        print(rec)

if __name__ == "__main__":
    main()