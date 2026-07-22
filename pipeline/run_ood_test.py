import json
import os
import sys
import copy
import pandas as pd

sys.path.insert(0, "src")

from engine import SimulationEngine
from schedulers import fcfs_scheduler, ltr_scheduler, robust_scheduler
from shared_structures import RequestPacket
from metrics import compute_metrics

OOD_LEVELS = [0, 20, 40, 60, 80]
BPS = 9.0  # normal scenario
MAX_TIME = 2000.0
MAX_WAIT = 300.0

# robust runs at the normal scenario tuned point
ROBUST_ALPHA = 0.25
ROBUST_BETA = 18.0

RESULTS_DIR = "results/ood_test"


def load_workload(ood_pct):
    with open(f"data/workload_ood_{ood_pct}.json") as f:
        raw = json.load(f)
    return [RequestPacket(**r) for r in raw]


def run_scheduler(requests, scheduler_fn, alpha, beta):
    engine = SimulationEngine(
        copy.deepcopy(requests),
        blocks_per_second=BPS,
        max_wait=MAX_WAIT,
    )
    engine.run(scheduler_fn=scheduler_fn, max_time=MAX_TIME, alpha=alpha, beta=beta)
    return engine.records


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    schedulers = [
        ("fcfs", fcfs_scheduler, 0, 0),
        ("ltr", ltr_scheduler, 0, 0),
        ("robust", robust_scheduler, ROBUST_ALPHA, ROBUST_BETA),
    ]

    all_results = {name: [] for name, _, _, _ in schedulers}

    for ood_pct in OOD_LEVELS:
        print(f"\nood level: {ood_pct}%")
        requests = load_workload(ood_pct)

        for name, fn, alpha, beta in schedulers:
            records = run_scheduler(requests, fn, alpha, beta)
            row = compute_metrics(records, ood_pct)
            all_results[name].append(row)
            print(f"  {name:8s} - jct: {row['jct']:.2f}s  starvation: {row['starvation_pct']:.1f}%  timed_out: {row['timed_out_pct']:.1f}%")

    for name, rows in all_results.items():
        path = os.path.join(RESULTS_DIR, f"results_{name}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
