import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

adv = pd.read_csv("results/advantage_over_fcfs.csv")
order = ["light", "normal", "stress", "saturation"]
adv["scenario"] = pd.Categorical(adv["scenario"], categories=order, ordered=True)
adv = adv.sort_values("scenario")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(adv["scenario"], adv["advantage_over_fcfs_pct"], color="#1D9E75")

for bar, val in zip(bars, adv["advantage_over_fcfs_pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha="center")

ax.set_ylabel("JCT advantage over FCFS (%)")
ax.set_xlabel("scenario (increasing server utilization)")
ax.set_title("robust scheduler advantage over FCFS grows with load")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("scripts/fig6_advantage_vs_fcfs.png", dpi=200)
print("saved scripts/fig6_advantage_vs_fcfs.png")
