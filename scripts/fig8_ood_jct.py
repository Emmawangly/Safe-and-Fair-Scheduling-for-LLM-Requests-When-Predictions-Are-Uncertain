import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

fcfs = pd.read_csv("results/ood_test/results_fcfs.csv").sort_values("error_pct")
ltr = pd.read_csv("results/ood_test/results_ltr.csv").sort_values("error_pct")
robust = pd.read_csv("results/ood_test/results_robust.csv").sort_values("error_pct")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fcfs["error_pct"], fcfs["jct"], marker="o", label="FCFS", color="#E24B4A")
ax.plot(ltr["error_pct"], ltr["jct"], marker="o", label="LTR", color="#378ADD")
ax.plot(robust["error_pct"], robust["jct"], marker="o", label="Robust (tuned)", color="#1D9E75")

ax.set_xlabel("out-of-distribution requests (%)")
ax.set_ylabel("mean job completion time (s)")
ax.set_title("JCT under out-of-distribution prediction failure")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
ax.grid(alpha=0.3)

fig.savefig("scripts/fig8_ood_jct.png", dpi=200, bbox_inches="tight")
print("saved scripts/fig8_ood_jct.png")
