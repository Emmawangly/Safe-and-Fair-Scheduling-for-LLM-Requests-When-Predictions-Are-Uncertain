import pandas as pd
import numpy as np


def compute_metrics(records, error_pct):
    if not records:
        raise ValueError("records list is empty.")

    required = {"arrival_time", "completion_time", "ttft", "preemptions"}
    missing = required - set(records[0].keys())
    if missing:
        raise ValueError(
            f"Engine records are missing required fields: {missing}. "
            "Make sure engine.py records ttft and preemptions per request."
        )

    df = pd.DataFrame(records)

    if "timed_out" in df.columns:
        timed_out_pct = float(df["timed_out"].mean() * 100)
        completed = df[~df["timed_out"]]
    else:
        timed_out_pct = 0.0
        completed = df

    # jct fixed with only requests that actually finishe

    jct = (completed["completion_time"] - completed["arrival_time"]).values
    n = len(jct)

    avg_jct = float(jct.mean()) if n > 0 else float("nan")
    avg_ttft = float(completed["ttft"].mean()) if n > 0 else float("nan")

    sum_x = jct.sum()
    sum_x2 = (jct ** 2).sum()

    jain = float((sum_x ** 2) / (n * sum_x2)) if sum_x2 > 0 else 1.0
    starved = float((jct > 3.0 * avg_jct).mean() * 100) if n > 0 else 0.0

    p95 = float(np.percentile(jct, 95)) if n > 0 else float("nan")
    p99 = float(np.percentile(jct, 99)) if n > 0 else float("nan")

    return {
        "error_pct": error_pct,
        "jct": round(avg_jct, 4),
        "ttft": round(avg_ttft, 4),
        "preemptions": int(df["preemptions"].sum()),
        "jain_fairness": round(jain, 4),
        "starvation_pct": round(starved, 2),
        "timed_out_pct": round(timed_out_pct, 2),
        "jct_p95": round(p95, 4),
        "jct_p99": round(p99, 4),
    }


def compute_metrics_batch(records, error_pct, alpha, beta):
    row = compute_metrics(records, error_pct)
    row["alpha"] = alpha
    row["beta"] = beta
    return row


if __name__ == "__main__":
    test_records = [
        {"arrival_time": 0.0, "completion_time": 2.5, "ttft": 0.0, "preemptions": 0, "timed_out": False},
        {"arrival_time": 0.5, "completion_time": 4.0, "ttft": 1.5, "preemptions": 1, "timed_out": False},
        {"arrival_time": 1.0, "completion_time": 6.0, "ttft": 2.0, "preemptions": 0, "timed_out": False},
        {"arrival_time": 1.5, "completion_time": 3.5, "ttft": 0.5, "preemptions": 0, "timed_out": False},
        {"arrival_time": 2.0, "completion_time": 15.0, "ttft": 4.0, "preemptions": 2, "timed_out": True},
    ]

    result = compute_metrics(test_records, error_pct=40)
    print("compute_metrics result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # jct now only averages the 4 requests that actually finished
    expected_jct = round((2.5 + 3.5 + 5.0 + 2.0) / 4, 4)
    assert result["jct"] == expected_jct
    print(f"\n  ✓ JCT ({result['jct']}s)")

    expected_ttft = round((0.0 + 1.5 + 2.0 + 0.5) / 4, 4)
    assert result["ttft"] == expected_ttft
    print(f"  ✓ TTFT ({result['ttft']}s)")

    assert result["preemptions"] == 3
    print(f"  ✓ Preemptions ({result['preemptions']})")

    assert result["starvation_pct"] == 0.0
    print(f"  ✓ Starvation ({result['starvation_pct']}%)")

    assert 0.0 <= result["jain_fairness"] <= 1.0
    print(f"  ✓ Jain's Fairness ({result['jain_fairness']})")

    assert result["timed_out_pct"] == 20.0
    print(f"  ✓ Timed out ({result['timed_out_pct']}%)")

    assert result["jct_p95"] >= result["jct"]
    print(f"  ✓ P95 ({result['jct_p95']}s)")

    assert result["jct_p99"] >= result["jct_p95"]
    print(f"  ✓ P99 ({result['jct_p99']}s)")

    try:
        compute_metrics([], error_pct=40)
        assert False
    except ValueError:
        print("  ✓ Empty records raises ValueError")

    try:
        compute_metrics([{"arrival_time": 0.0}], error_pct=40)
        assert False
    except ValueError as e:
        print(f"  ✓ Missing fields raises ValueError")

    no_timeout = [{"arrival_time": 0.0, "completion_time": 3.0, "ttft": 0.5, "preemptions": 0}]
    r2 = compute_metrics(no_timeout, error_pct=0)
    assert r2["timed_out_pct"] == 0.0
    print("  ✓ timed_out_pct = 0.0 when field absent")

    batch = compute_metrics_batch(test_records, error_pct=40, alpha=0.5, beta=0.5)
    assert "alpha" in batch and "jct_p95" in batch
    print("  ✓ compute_metrics_batch ok")

    print("\n✓ all tests passed ✓")
