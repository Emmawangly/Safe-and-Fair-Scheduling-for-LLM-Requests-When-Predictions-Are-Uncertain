import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# shows jct across every alpha/beta pair we tested, star marks the best one
plt.rcParams.update({"font.size": 10})

scenario = "normal"
error_level = 40

sweep = pd.read_csv("results/results_parameter_sweep.csv")
subset = sweep[(sweep["scenario"] == scenario) & (sweep["error_pct"] == error_level)]

alphas = sorted(subset["alpha"].unique())
betas = sorted(subset["beta"].unique())

grid = np.zeros((len(alphas), len(betas)))
for i, a in enumerate(alphas):
    for j, b in enumerate(betas):
        match = subset[(subset["alpha"] == a) & (subset["beta"] == b)]
        grid[i, j] = match["jct"].iloc[0]

best_i, best_j = np.unravel_index(np.argmin(grid), grid.shape)
vmin, vmax = grid.min(), grid.max()

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(betas)))
ax.set_xticklabels([str(b) for b in betas], rotation=45)
ax.set_yticks(range(len(alphas)))
ax.set_yticklabels([str(a) for a in alphas])
ax.set_xlabel("beta")
ax.set_ylabel("alpha")
ax.set_title(f"mean JCT across alpha/beta - {scenario} scenario, {error_level}% error")

for i in range(len(alphas)):
    for j in range(len(betas)):
        val = grid[i, j]
        normalized = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
        text_color = "white" if normalized > 0.55 else "black"
        is_best = (i == best_i and j == best_j)
        label = ("*" + f"{val:.0f}") if is_best else f"{val:.0f}"
        ax.text(j, i, label, ha="center", va="center", color=text_color,
                fontsize=7, fontweight="bold" if is_best else "normal")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("mean JCT (s)")

fig.tight_layout()
fig.savefig("scripts/fig3_heatmap.png", dpi=200)
print("saved scripts/fig3_heatmap.png")
