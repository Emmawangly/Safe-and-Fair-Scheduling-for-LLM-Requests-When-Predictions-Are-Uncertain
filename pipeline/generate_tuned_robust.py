import json
import sys
import pandas as pd

sys.path.insert(0, "src")

from engine import SimulationEngine
from schedulers import robust_scheduler
from shared_structures import RequestPacket
from metrics import compute_metrics

ERROR_LEVELS = [0, 20, 40, 60, 80]

MAX_TIME = 2000.0
MAX_WAIT = 300.0

# same  values as sweep 
scenario_bps = {"light": 15.8, "normal": 9.0, "stress": 4.5, "saturation": 3.2}


best_params = pd.read_csv("results/best_params_per_scenario.csv")


def load_workload(error_pct):
    with open(f"data/workload_error_{error_pct}.json") as f:
        raw = json.load(f)
    return [RequestPacket(**r) for r in raw]


for _, row in best_params.iterrows():
    scenario = row["scenario"]
    alpha = row["alpha"]
    beta = row["beta"]
    bps = scenario_bps[scenario]

    print(f"{scenario}: tuned alpha={alpha}, beta={beta}")
    rows = []
    for error_pct in ERROR_LEVELS:
        requests = load_workload(error_pct)
        engine = SimulationEngine(list(requests), blocks_per_second=bps, max_wait=MAX_WAIT)
        engine.run(scheduler_fn=robust_scheduler, max_time=MAX_TIME, alpha=alpha, beta=beta)
        metrics_row = compute_metrics(engine.records, error_pct)
        rows.append(metrics_row)

    out_path = f"results/{scenario}/results_robust_tuned.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  saved {out_path}")
