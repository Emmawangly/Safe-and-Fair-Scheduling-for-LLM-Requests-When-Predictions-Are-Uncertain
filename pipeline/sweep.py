import os
import sys
import json
import copy
import pandas as pd

sys.path.insert(0, "src")

from engine import SimulationEngine
from schedulers import robust_scheduler
from shared_structures import RequestPacket
from metrics import compute_metrics_batch

ERROR_LEVELS = [0, 20, 40, 60, 80]

#false=full sweep, true=test first
QUICK_TEST = False

if QUICK_TEST:
    ALPHAS = [0.5, 1.0]
    BETAS = [0.5, 1.0]
    SCENARIOS = {"normal": 9.0}
else:
    ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
   
   #new beta values included for better starvation
    BETAS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 5.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 25.00, 30.00]
    SCENARIOS = {
        "light": 15.8,
        "normal": 9.0,
        "stress": 4.5,
        "saturation": 3.2,
    }

TIME_STEP = 1.0
MAX_TIME = 2000.0
MAX_WAIT = 300.0

RESULTS_DIR = "results"


def load_workload(error_pct):
    path = f"data/workload_error_{error_pct}.json"

    with open(path) as f:
        raw = json.load(f)

    return [RequestPacket(**r) for r in raw]


def run_scheduler(requests, alpha, beta, blocks_per_second):
    
    engine = SimulationEngine(
        copy.deepcopy(requests),
        time_step=TIME_STEP,
        blocks_per_second=blocks_per_second,
        max_wait=MAX_WAIT,
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
    total = len(SCENARIOS) * len(ERROR_LEVELS) * len(ALPHAS) * len(BETAS)
    done = 0

    for scenario_name, bps in SCENARIOS.items():
        for error_pct in ERROR_LEVELS:
            requests = load_workload(error_pct)

            for alpha in ALPHAS:
                for beta in BETAS:
                    records = run_scheduler(requests, alpha, beta, bps)

                    row = compute_metrics_batch(
                        records,
                        error_pct,
                        alpha,
                        beta,
                    )

                    row["scenario"] = scenario_name
                    parameter_rows.append(row)

                    done += 1
                    if done % 20 == 0:
                        print(f"progress: {done}/{total}")

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

    # creates the best parameters for alpha and beta
    best_params_path = os.path.join(RESULTS_DIR, "best_params_per_scenario.csv")
    try:
        best_params = pd.read_csv(best_params_path)
        normal_row = best_params[best_params["scenario"] == "normal"].iloc[0]
        tuned_alpha = float(normal_row["alpha"])
        tuned_beta = float(normal_row["beta"])
        print(f"ablation baseline (normal, from best_params_per_scenario.csv): alpha={tuned_alpha}, beta={tuned_beta}")
    except (FileNotFoundError, IndexError):
         
        print("best_params_per_scenario.csv not found, run select_best_params.py first for the real tuned point")
        print("falling back to an inline estimate for now")
        sweep_so_far = pd.read_csv(os.path.join(RESULTS_DIR, "results_parameter_sweep.csv"))
        normal = sweep_so_far[sweep_so_far["scenario"] == "normal"]
        grouped = normal.groupby(["alpha", "beta"])
        stats = [{"alpha": a, "beta": b, "worst_starvation": g["starvation_pct"].max(), "avg_jct": g["jct"].mean()}
                  for (a, b), g in grouped if len(g) == len(ERROR_LEVELS)]
        stats_df = pd.DataFrame(stats)
        safe = stats_df[stats_df["worst_starvation"] <= 2.0]
        chosen = safe.sort_values("avg_jct").iloc[0] if not safe.empty else stats_df.sort_values("worst_starvation").iloc[0]
        tuned_alpha, tuned_beta = float(chosen["alpha"]), float(chosen["beta"])
        print(f"ablation baseline (normal, fallback estimate): alpha={tuned_alpha}, beta={tuned_beta}")

    ablation_configs = [
        ("full", tuned_alpha, tuned_beta),
        ("no_uncertainty", 0.0, tuned_beta),
        ("no_aging", tuned_alpha, 0.0),
    ]

    normal_bps = SCENARIOS.get("normal", 9.0)

    for name, alpha, beta in ablation_configs:
        print(f"\nRunning ablation: {name}")

        for error_pct in ERROR_LEVELS:
            requests = load_workload(error_pct)

            records = run_scheduler(requests, alpha, beta, normal_bps)

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
    print("\nsweep done. run select_best_params.py, then sweep.run_ablation() separately")
    print("(ablation needs best_params_per_scenario.csv, which select_best_params.py produces)")


if __name__ == "__main__":
    main()