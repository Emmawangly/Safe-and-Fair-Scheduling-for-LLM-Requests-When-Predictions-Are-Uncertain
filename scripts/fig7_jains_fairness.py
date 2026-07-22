import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

scenario = "normal"
folder = f"results/{scenario}"

fcfs = pd.read_csv(f"{folder}/results_fcfs.csv").sort_values("error_pct")
ltr = pd.read_csv(f"{folder}/results_ltr.csv").sort_values("error_pct")
robust = pd.read_csv(f"{folder}/results_robust_tuned.csv").sort_values("error_pct")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fcfs["error_pct"], fcfs["jain_fairness"], marker="o", label="FCFS", color="#E24B4A")
ax.plot(ltr["error_pct"], ltr["jain_fairness"], marker="o", label="LTR", color="#378ADD")
ax.plot(robust["error_pct"], robust["jain_fairness"], marker="o", label="Robust (tuned)", color="#1D9E75")

ax.set_xlabel("prediction error (%)")
ax.set_ylabel("jain's fairness index")
ax.set_ylim(0, 1.05)
ax.set_title("fairness vs prediction error - Normal Load scenario")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
ax.grid(alpha=0.3)

fig.savefig("scripts/fig7_jains_fairness.png", dpi=200, bbox_inches="tight")
print("saved scripts/fig7_jains_fairness.png")
