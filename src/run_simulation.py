import json
import os
import copy
import pandas as pd

from engine import SimulationEngine
from schedulers import fcfs_scheduler, ltr_scheduler, robust_scheduler
from ljf import ljf_scheduler
from shared_structures import RequestPacket
from metrics import compute_metrics

ERROR_LEVELS = [0, 20, 40, 60, 80]

# untuned default, robust also gets run again per scenario with tuned alpha/beta separately

ALPHA = 0.5
BETA = 0.5

TIME_STEP = 1.0
MAX_TIME = 2000.0
MAX_WAIT = 300.0
RESULTS_DIR = "results"


SCENARIOS = {
    "light": 15.8,
    "normal": 9.0,
    "stress": 4.5,
    "saturation": 3.2,
}


def load_workload(error_pct):
    path = f"data/workload_error_{error_pct}.json"
    with open(path) as f:
        raw = json.load(f)
    return [RequestPacket(**r) for r in raw]


def run_scheduler(requests, scheduler_fn, alpha, beta, blocks_per_second):
    
    
    engine = SimulationEngine(
        copy.deepcopy(requests),
        time_step=TIME_STEP,
        blocks_per_second=blocks_per_second,
        max_wait=MAX_WAIT,
    )
    engine.run(
        scheduler_fn=scheduler_fn,
        max_time=MAX_TIME,
        alpha=alpha,
        beta=beta,
    )
    return engine.records


def main():
    schedulers = [
        ("fcfs", fcfs_scheduler),
        ("ltr", ltr_scheduler),
        ("robust", robust_scheduler),
        ("ljf", ljf_scheduler),
    ]

    for scenario_name, bps in SCENARIOS.items():
        print(f"\n=== Scenario: {scenario_name} (blocks_per_second={bps}) ===")
        scenario_dir = os.path.join(RESULTS_DIR, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        all_results = {name: [] for name, _ in schedulers}

        for error_pct in ERROR_LEVELS:
            print(f"\nError level: {error_pct}%")
            requests = load_workload(error_pct)

            for name, fn in schedulers:
                records = run_scheduler(requests, fn, ALPHA, BETA, bps)
                row = compute_metrics(records, error_pct)
                all_results[name].append(row)
                print(f"  {name:8s} - JCT: {row['jct']:.3f}s  "
                      f"TTFT: {row['ttft']:.3f}s  "
                      f"preemptions: {row['preemptions']}  "
                      f"jain: {row['jain_fairness']:.3f}  "
                      f"starvation: {row['starvation_pct']:.1f}%  "
                      f"timed_out: {row['timed_out_pct']:.1f}%")

        for name, rows in all_results.items():
            path = os.path.join(scenario_dir, f"results_{name}.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"Saved {path}")

    print("\nAll scenarios complete ✓")


if __name__ == "__main__":
    main()