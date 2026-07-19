import pandas as pd
import numpy as np
from typing import List


def compute_metrics(records: List[dict], error_pct: int) -> dict:
   
    if not records:
        raise ValueError("records list is empty — no data to aggregate.")

    required_fields = {"arrival_time", "completion_time", "ttft", "preemptions"}
    missing = required_fields - set(records[0].keys())
    if missing:
        raise ValueError(
            f"Engine records are missing required fields: {missing}. "
            f"Make sure engine.py records ttft and preemptions per request."
        )

    df = pd.DataFrame(records)

    # ── Job Completion Time ───────────────────────────────────────────────────
    # JCT = time from arrival until the request fully completes
    jct_vals = (df["completion_time"] - df["arrival_time"]).values
    mean_jct = float(jct_vals.mean())

    # ── Time to First Token ───────────────────────────────────────────────────
    # TTFT = wait_time at the moment the request was selected for execution
    # Recorded by the engine when _start_request() is called
    mean_ttft = float(df["ttft"].mean())

    # ── Preemptions ───────────────────────────────────────────────────────────
    # Total number of preemption events across all requests in this run
    total_preemptions = int(df["preemptions"].sum())

    # ── Jain's Fairness Index ─────────────────────────────────────────────────
    # Formula: J = (Σ xᵢ)² / (n · Σ xᵢ²)
    # Range: [1/n, 1.0] — closer to 1.0 means fairer distribution of JCT
    n = len(jct_vals)
    sum_jct    = jct_vals.sum()
    sum_jct_sq = (jct_vals ** 2).sum()

    if sum_jct_sq == 0:
        jain = 1.0  # all JCTs are 0 — perfectly fair (edge case)
    else:
        jain = float((sum_jct ** 2) / (n * sum_jct_sq))

    # ── Starvation frequency ──────────────────────────────────────────────────
    # A request is considered starved if its JCT exceeds 3x the mean JCT.
    # Threshold of 3x follows the convention used in the project proposal.
    starvation_threshold = 3.0 * mean_jct
    starvation_pct = float((jct_vals > starvation_threshold).mean() * 100)

    return {
        "error_pct":      error_pct,
        "jct":            round(mean_jct, 4),
        "ttft":           round(mean_ttft, 4),
        "preemptions":    total_preemptions,
        "jain_fairness":  round(jain, 4),
        "starvation_pct": round(starvation_pct, 2),
    }


def compute_metrics_batch(
    records: List[dict],
    error_pct: int,
    alpha: float,
    beta: float,
) -> dict:
    """
    Extended version of compute_metrics that also records the α and β
    parameters used in the experiment. Useful for the heatmap sweep.

    Args:
        records:   Same as compute_metrics.
        error_pct: Same as compute_metrics.
        alpha:     Uncertainty penalty weight used in the Robust scheduler.
        beta:      Anti-starvation aging weight used in the Robust scheduler.

    Returns:
        Same dict as compute_metrics, plus keys: alpha, beta.
    """
    row = compute_metrics(records, error_pct)
    row["alpha"] = alpha
    row["beta"]  = beta
    return row


if __name__ == "__main__":
    # ── Self-test with synthetic records ─────────────────────────────────────
    synthetic_records = [
        {"arrival_time": 0.0, "completion_time": 2.5,  "ttft": 0.0, "preemptions": 0},
        {"arrival_time": 0.5, "completion_time": 4.0,  "ttft": 1.5, "preemptions": 1},
        {"arrival_time": 1.0, "completion_time": 6.0,  "ttft": 2.0, "preemptions": 0},
        {"arrival_time": 1.5, "completion_time": 3.5,  "ttft": 0.5, "preemptions": 0},
        {"arrival_time": 2.0, "completion_time": 15.0, "ttft": 4.0, "preemptions": 2},
    ]

    result = compute_metrics(synthetic_records, error_pct=40)
    print("compute_metrics output:")
    for key, val in result.items():
        print(f"  {key}: {val}")

    # Verify expected values
    # JCT values: 2.5, 3.5, 5.0, 2.0, 13.0 → mean = 5.2
    expected_jct = round((2.5 + 3.5 + 5.0 + 2.0 + 13.0) / 5, 4)
    assert result["jct"] == expected_jct, f"JCT mismatch: {result['jct']} != {expected_jct}"
    print(f"\n✅ JCT = {result['jct']}s (expected {expected_jct}s)")

    # TTFT mean: (0.0+1.5+2.0+0.5+4.0)/5 = 1.6
    expected_ttft = round((0.0 + 1.5 + 2.0 + 0.5 + 4.0) / 5, 4)
    assert result["ttft"] == expected_ttft, f"TTFT mismatch"
    print(f"✅ TTFT = {result['ttft']}s (expected {expected_ttft}s)")

    # Preemptions: 0+1+0+0+2 = 3
    assert result["preemptions"] == 3
    print(f"✅ Preemptions = {result['preemptions']} (expected 3)")

    # Jain's: request 5 (JCT=13.0) is >> 3x mean (3*5.2=15.6) → not starved
    # Actually 13.0 < 15.6 so starvation should be 0%
    assert result["starvation_pct"] == 0.0, f"Starvation: {result['starvation_pct']}"
    print(f"✅ Starvation = {result['starvation_pct']}% (expected 0.0%)")

    # Jain's range check
    assert 0.0 <= result["jain_fairness"] <= 1.0
    print(f"✅ Jain's Fairness = {result['jain_fairness']} (valid range [0,1])")

    # Edge case: empty records
    try:
        compute_metrics([], error_pct=40)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Empty records raises ValueError")

    # Edge case: missing fields
    try:
        compute_metrics([{"arrival_time": 0.0}], error_pct=40)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Missing fields raises ValueError: {e}")

    # compute_metrics_batch
    row = compute_metrics_batch(synthetic_records, error_pct=40, alpha=0.5, beta=0.5)
    assert row["alpha"] == 0.5 and row["beta"] == 0.5
    print("✅ compute_metrics_batch includes alpha and beta")

    print("\nAll tests passed.")