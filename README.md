# Safe and Fair Scheduling for LLM Requests When Predictions Are Uncertain

CSCI 6806 / INFO 4205 Capstone Project

This project develops a Python discrete-event simulation framework for comparing LLM request scheduling policies under uncertain resource predictions.

The project compares:

* First-Come-First-Served (FCFS)
* Learning-to-Rank (LTR) scheduling
* A Robust Scheduler that considers predicted cost, uncertainty, and waiting time

## Member 1: Workload Generator

The workload generator is located at:

```text
src/workload_generator.py
```

It creates synthetic LLM request packets for simulation experiments.

Each request contains the shared interface fields:

```python
{
    "request_id": int,
    "arrival_time": float,
    "actual_blocks": int,
    "predicted_mu": float,
    "predicted_sigma": float,
    "wait_time": float,
}
```

### Field meanings

| Field             | Meaning                                           | Used by           |
| ----------------- | ------------------------------------------------- | ----------------- |
| `request_id`      | Unique request identifier                         | All modules       |
| `arrival_time`    | Simulated time when the request enters the system | Engine            |
| `actual_blocks`   | True KV-cache block requirement                   | Engine only       |
| `predicted_mu`    | Predicted memory cost in KV-cache blocks          | Scheduler         |
| `predicted_sigma` | Prediction uncertainty                            | Robust Scheduler  |
| `wait_time`       | Time currently spent waiting in the queue         | Updated by Engine |

The scheduler must not use `actual_blocks` when making scheduling decisions. It should use only predicted values such as `predicted_mu` and `predicted_sigma`.

## Generate Workloads

Run the generator from the project root:

```bash
python src/workload_generator.py
```

This creates five workload files in the `data/` folder:

```text
data/workload_error_0.json
data/workload_error_20.json
data/workload_error_40.json
data/workload_error_60.json
data/workload_error_80.json
```

Each workload contains 1,000 simulated requests.

The error levels represent prediction error rates of:

```text
0%, 20%, 40%, 60%, and 80%
```

A small integration sample is also created:

```text
sample_output.json
```

This file contains 10 requests with an 80% prediction error rate.

## Use the Generator in Python

Other modules can import the generator directly:

```python
from workload_generator import generate_workload

queue = generate_workload(
    num_requests=1000,
    error_rate_percent=40,
    seed=42,
)
```

Using the same seed produces the same workload, which supports fair comparisons between scheduling algorithms.


## Install Dependencies

Create and activate a Python virtual environment, then install project dependencies:

```bash
python -m pip install -r requirements.txt


## Current Status

The current workload generator uses synthetic request lengths and configurable prediction noise.

The next planned update is to replace the temporary synthetic block-length distribution with a distribution derived from LMSYS-Chat-1M data, while keeping the same shared request interface.
## LMSYS-Style Profile Pipeline

The repository includes a reusable pipeline for converting LMSYS-style conversation data into workload profiles.

Pipeline:

```text
JSONL conversation records
→ estimated token lengths
→ KV-cache block profile
→ profile-based workload generation
```

### Build a Block Profile

The extractor expects a local JSONL file with conversation records in this format:

```json
{
  "conversation": [
    {"role": "user", "content": "Example user request"},
    {"role": "assistant", "content": "Example assistant response"}
  ]
}
```

Run:

```bash
python src/lmsys_extract_profile.py
```

This creates:

```text
data/lmsys_block_profile.json
```

### Generate Profile-Based Workloads

After a profile is available, run:

```bash
python src/generate_profile_workloads.py
```

The script generates workloads for 0%, 20%, 40%, 60%, and 80% prediction error and verifies that all error levels use identical `actual_blocks` values.

### Important Note

The current `lmsys_sample.jsonl` file is a small local example used only to validate the pipeline.

It is not the real LMSYS-Chat-1M dataset.

The current token estimator uses approximately four characters per token. Before final experiments, this should be replaced or validated with the tokenizer selected by the project team.
