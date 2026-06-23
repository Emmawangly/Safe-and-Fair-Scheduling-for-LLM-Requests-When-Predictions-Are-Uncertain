import json
import os
import pandas as pd

from workload_generator import generate_workload
from engine import SimulationEngine
from schedulers import fcfs_scheduler, ltr_scheduler, robust_scheduler
from shared_structures import RequestPacket
from metrics import compute_metrics, compute_metrics_batch

ERROR_LEVELS  = [0, 20, 40, 60, 80]
NUM_REQUESTS  = 1000
SEED          = 42
ALPHA         = 0.5
BETA          = 0.5
TIME_STEP     = 1.0
BLOCKS_PER_S  = 4
MAX_TIME      = 2000.0
RESULTS_DIR   = "results"


def load_workload(error_pct: int) -> list:
    path = f"data/workload_error_{error_pct}.json"
    with open(path) as f:
        raw = json.load(f)
    return [RequestPacket(**r) for r in raw]


def run_scheduler(requests: list, scheduler_fn, alpha: float, beta: float) -> list:
    engine = SimulationEngine(
        list(requests),
        time_step=TIME_STEP,
        blocks_per_second=BLOCKS_PER_S,
    )
    engine.run(
        scheduler_fn=scheduler_fn,
        max_time=MAX_TIME,
        alpha=alpha,
        beta=beta,
    )
    return engine.records


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    schedulers = [
        ("fcfs",   fcfs_scheduler),
        ("ltr",    ltr_scheduler),
        ("robust", robust_scheduler),
    ]

    all_results = {name: [] for name, _ in schedulers}

    for error_pct in ERROR_LEVELS:
        print(f"\nError level: {error_pct}%")
        requests = load_workload(error_pct)

        for name, fn in schedulers:
            records = run_scheduler(requests, fn, ALPHA, BETA)
            row = compute_metrics(records, error_pct)
            all_results[name].append(row)
            print(f"  {name:8s} — JCT: {row['jct']:.3f}s  "
                  f"TTFT: {row['ttft']:.3f}s  "
                  f"Preemptions: {row['preemptions']}  "
                  f"Jain: {row['jain_fairness']:.3f}  "
                  f"Starvation: {row['starvation_pct']:.1f}%")

    for name, rows in all_results.items():
        path = os.path.join(RESULTS_DIR, f"results_{name}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\nSaved {path}")

    print("\nAll experiments complete.")
    print("Connect the dashboard by replacing load_data() with:")
    print('  "fcfs":   pd.read_csv("results/results_fcfs.csv")')
    print('  "ltr":    pd.read_csv("results/results_ltr.csv")')
    print('  "robust": pd.read_csv("results/results_robust.csv")')


if __name__ == "__main__":
    main()