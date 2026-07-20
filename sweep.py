import os
import json
import pandas as pd

from src.engine import SimulationEngine
from src.schedulers import robust_scheduler
from src.shared_structures import RequestPacket
from metrics import compute_metrics_batch
ERROR_LEVELS = [0, 20, 40, 60, 80]

ALPHAS = [0.1, 0.5, 1.0, 2.0]
BETAS  = [0.1, 0.5, 1.0, 2.0]

TIME_STEP = 1.0
BLOCKS_PER_S = 4
MAX_TIME = 2000.0

RESULTS_DIR = "results"
def load_workload(error_pct):
    path = f"data/workload_error_{error_pct}.json"

    with open(path) as f:
        raw = json.load(f)

    return [RequestPacket(**r) for r in raw]
def run_scheduler(requests, alpha, beta):
    engine = SimulationEngine(
        list(requests),
        time_step=TIME_STEP,
        blocks_per_second=BLOCKS_PER_S,
    )

    engine.run(
        scheduler_fn=robust_scheduler,
        max_time=MAX_TIME,
        alpha=alpha,
        beta=beta,
    )

    return engine.records
def run_parameter_sweep():
    parameter_rows = []

    for alpha in ALPHAS:
        for beta in BETAS:
            print(f"\nRunning alpha={alpha}, beta={beta}")

            for error_pct in ERROR_LEVELS:
                requests = load_workload(error_pct)

                records = run_scheduler(
                    requests,
                    alpha,
                    beta,
                )

                row = compute_metrics_batch(
                    records,
                    error_pct,
                    alpha,
                    beta,
                )

                parameter_rows.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULTS_DIR,
        "results_parameter_sweep.csv",
    )

    pd.DataFrame(parameter_rows).to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {output_path}")
def run_ablation():
    ablation_rows = []

    ablation_configs = [
        ("full", 0.5, 0.5),
        ("no_uncertainty", 0.0, 0.5),
        ("no_aging", 0.5, 0.0),
    ]

    for name, alpha, beta in ablation_configs:
        print(f"\nRunning ablation: {name}")

        for error_pct in ERROR_LEVELS:
            requests = load_workload(error_pct)

            records = run_scheduler(
                requests,
                alpha,
                beta,
            )

            row = compute_metrics_batch(
                records,
                error_pct,
                alpha,
                beta,
            )

            row["configuration"] = name
            ablation_rows.append(row)

    output_path = os.path.join(
        RESULTS_DIR,
        "results_ablation.csv",
    )

    pd.DataFrame(ablation_rows).to_csv(
        output_path,
        index=False,
    )

    print(f"Saved {output_path}")
def main():
    run_parameter_sweep()
    run_ablation()


if __name__ == "__main__":
    main()
