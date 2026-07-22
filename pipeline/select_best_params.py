import pandas as pd

STARVATION_LIMIT = 2.0  
ERROR_LEVELS = [0, 20, 40, 60, 80]
ANCHOR_ERROR = 40  # for the pareto curve and the sweep row

sweep = pd.read_csv("results/results_parameter_sweep.csv")
scenarios = sweep["scenario"].unique()



best_rows = []
for scenario in scenarios:
    subset = sweep[sweep["scenario"] == scenario]
    grouped = subset.groupby(["alpha", "beta"])

    stats = []
    for (alpha, beta), g in grouped:
        if len(g) < len(ERROR_LEVELS):
            continue
        stats.append({
            "alpha": alpha,
            "beta": beta,
            "worst_starvation": g["starvation_pct"].max(),
            "avg_jct": g["jct"].mean(),
        })
    stats_df = pd.DataFrame(stats)

    safe = stats_df[stats_df["worst_starvation"] <= STARVATION_LIMIT]
    chosen = safe.sort_values("avg_jct").iloc[0] if not safe.empty else stats_df.sort_values("worst_starvation").iloc[0]

    row = subset[
        (subset["alpha"].round(3) == round(chosen["alpha"], 3))
        & (subset["beta"].round(3) == round(chosen["beta"], 3))
        & (subset["error_pct"] == ANCHOR_ERROR)
    ].iloc[0].copy()
    row["worst_case_starvation"] = chosen["worst_starvation"]
    row["avg_jct_across_errors"] = chosen["avg_jct"]
    best_rows.append(row)

best_df = pd.DataFrame(best_rows)
best_df.to_csv("results/best_params_per_scenario.csv", index=False)
print(f"best (alpha, beta) per scenario, robust across all 5 error levels (starvation <= {STARVATION_LIMIT}% everywhere):")
print(best_df[["scenario", "alpha", "beta", "avg_jct_across_errors", "worst_case_starvation"]].to_string(index=False))

# pareto curve: 

pareto_rows = []
for scenario in scenarios:
    alpha_star = best_df[best_df["scenario"] == scenario]["alpha"].iloc[0]
    curve = sweep[
        (sweep["scenario"] == scenario)
        & (sweep["alpha"] == alpha_star)
        & (sweep["error_pct"] == ANCHOR_ERROR)
    ]
    pareto_rows.append(curve)

pareto_df = pd.concat(pareto_rows)
pareto_df.to_csv("results/pareto_curve.csv", index=False)
print(f"\nsaved results/pareto_curve.csv, {len(pareto_df)} rows")

# advantage over fcfs
rows = []
for scenario in scenarios:
    fcfs = pd.read_csv(f"results/{scenario}/results_fcfs.csv")
    fcfs_avg_jct = fcfs["jct"].mean()
    robust_row = best_df[best_df["scenario"] == scenario].iloc[0]
    advantage_pct = 100 * (1 - robust_row["avg_jct_across_errors"] / fcfs_avg_jct)
    rows.append({
        "scenario": scenario,
        "fcfs_avg_jct": fcfs_avg_jct,
        "robust_avg_jct": robust_row["avg_jct_across_errors"],
        "robust_worst_starvation": robust_row["worst_case_starvation"],
        "advantage_over_fcfs_pct": advantage_pct,
    })

advantage_df = pd.DataFrame(rows)
advantage_df.to_csv("results/advantage_over_fcfs.csv", index=False)
print("\nadvantage over fcfs, averaged across all 5 error levels:")
print(advantage_df.to_string(index=False))
