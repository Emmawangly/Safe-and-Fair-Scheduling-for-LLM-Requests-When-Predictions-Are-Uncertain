import pandas as pd
import matplotlib.pyplot as plt

# each line is one scenario, only the points where no other beta beats it
# on both jct and starvation at once -- that removes the zigzag from
# beta's non-monotonic effect (starvation gets worse before it gets better)
plt.rcParams.update({"font.size": 11})

pareto = pd.read_csv("results/pareto_curve.csv")
colors = {"light": "#27AE60", "normal": "#2874A7", "stress": "#BA4A00", "saturation": "#922B21"}


def pareto_frontier(df):
    df = df.sort_values("jct")
    frontier = []
    best_starvation = float("inf")
    for _, row in df.iterrows():
        if row["starvation_pct"] < best_starvation:
            frontier.append(row)
            best_starvation = row["starvation_pct"]
    return pd.DataFrame(frontier)


fig, ax = plt.subplots(figsize=(6.5, 5))

for scenario, color in colors.items():
    curve = pareto[pareto["scenario"] == scenario]
    frontier = pareto_frontier(curve).sort_values("jct")
    ax.plot(frontier["starvation_pct"], frontier["jct"], marker="o", label=scenario, color=color)

ax.set_xlabel("starvation rate (%)")
ax.set_ylabel("mean JCT (s)")
ax.set_title("JCT vs starvation efficient frontier (beta sweep)")
ax.legend(title="scenario", loc="upper right", framealpha=0.95)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("scripts/fig5_pareto.png", dpi=200)
print("saved scripts/fig5_pareto.png")
