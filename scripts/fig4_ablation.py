import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

error_level = 40

ablation = pd.read_csv("results/results_ablation.csv")
subset = ablation[ablation["error_pct"] == error_level]

labels = ["full formula", "alpha = 0", "beta = 0"]
configs = ["full", "no_uncertainty", "no_aging"]
jct_vals = [subset[subset["configuration"] == c]["jct"].iloc[0] for c in configs]
starv_vals = [subset[subset["configuration"] == c]["starvation_pct"].iloc[0] for c in configs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.bar(labels, jct_vals, color=["#1D9E75", "#9B59B6", "#F1C40F"])
ax1.set_ylabel("mean JCT (s)")
ax1.set_title(f"JCT at {error_level}% error")
ax1.tick_params(axis="x", rotation=15)

ax2.bar(labels, starv_vals, color=["#1D9E75", "#9B59B6", "#F1C40F"])
ax2.set_ylabel("starvation rate (%)")
ax2.set_title(f"starvation at {error_level}% error")
ax2.tick_params(axis="x", rotation=15)

fig.tight_layout()
fig.savefig("scripts/fig4_ablation.png", dpi=200)
print("saved scripts/fig4_ablation.png")
