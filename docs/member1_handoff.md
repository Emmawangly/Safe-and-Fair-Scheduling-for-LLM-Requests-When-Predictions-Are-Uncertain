# Member 1 Workload Generator Handoff

## Purpose

Member 1 provides reproducible LLM request workloads for the simulation engine.

The generator supports prediction error rates of:

```text
0%, 20%, 40%, 60%, and 80%
For fair scheduler comparisons, use the same seed and the same underlying workload across FCFS, LTR, and the Robust Scheduler.

Main Generator

Location:

src/workload_generator.py

Main function:

generate_workload(
    num_requests,
    error_rate_percent,
    seed=42,
    block_profile=None,
)

Example:

from workload_generator import generate_workload

queue = generate_workload(
    num_requests=1000,
    error_rate_percent=40,
    seed=42,
)
Request Packet Format

Each request is a dictionary with these six shared fields:

{
    "request_id": int,
    "arrival_time": float,
    "actual_blocks": int,
    "predicted_mu": float,
    "predicted_sigma": float,
    "wait_time": float,
}
Field Ownership
Field	Meaning	Owner / Usage
request_id	Unique request identifier	All modules
arrival_time	Simulated request arrival time	Engine
actual_blocks	True KV-cache block requirement	Engine only
predicted_mu	Predicted memory cost	Scheduler
predicted_sigma	Prediction uncertainty	Robust Scheduler
wait_time	Current queue waiting time	Updated by Engine
Important Integration Rules
The engine should admit requests according to arrival_time.
The engine should use actual_blocks for actual memory allocation and preemption decisions.
The scheduler must not use actual_blocks when deciding priority.
The scheduler should use only:
predicted_mu
predicted_sigma
wait_time
wait_time is initialized to 0.0 by Member 1 and must be updated by the engine during simulation.
For fair comparisons, use the same:
seed
num_requests
error_rate_percent
block_profile

for each scheduler experiment.

Small Integration Input

For a quick integration test, use:

sample_output.json

It contains 10 requests and is useful before running 1,000-request workloads.

Profile-Based Workloads

The project supports profile-based workload generation:

JSONL or Parquet conversation data
→ token-length estimation
→ KV-cache block profile
→ workload generation

Relevant files:

src/lmsys_extract_profile.py
src/lmsys_profile.py
src/generate_profile_workloads.py
Validation Status

Automated tests currently verify:

request field names and data types;
reproducibility with a fixed seed;
0% prediction-error behavior;
uncertainty behavior at higher error rates;
JSONL profile extraction;
Parquet profile extraction;
profile-based workload generation;
identical actual block requirements across error levels.

Run all tests with:

python -m pytest tests