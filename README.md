# Safe and Fair Scheduling for LLM Requests When Predictions Are Uncertain

CSCI 6806 Capstone Project — FDU Vancouver
Group members: Chang Liu, Juan Garcia, Liyang Wang, Loriynne Duarte

A Python discrete-event simulator created to compare LLM request scheduling policies under uncertain resource predictions

## Schedulers that were compared in the studies

**FCFS** —> first-come-first-served. Our baseline 

**LTR** —> learning-to-rank. Our starting point in the extension research

**LJF** —> longest-job-first. This was used mainly to have a parameter of the worst case scenario

**Robust** —> our scheduler! It combines a prediction-uncertainty penalty with an aging term:

```text
Robust Score Formula = T_wait_avg / (mu_i + alpha * sigma_i) + beta * (T_wait_i / T_wait_avg)
```

## Git Repository structure

```text
src/
engine, schedulers, metrics, workload generator

dashboard/
dashboard

pipeline/
sweep, parameter selection, OOD test

scripts/
matplotlib figures scripts used in the report 

tests/
automated unit tests

data/
generated workload files

results/
generated CSV output from every experiment 

docs/
fina reports

```

## Running the Project!

1. Before running the tests and the simulator, is necessary to install all the libraries to run the project. This command will install the libraries: 

```bash
pip install -r requirements.txt
```

2. Bellow is the order in which the tests must be executed. They must be done in order, because one output will feed the next one.

```bash
py -m pytest tests/ -v # this will run the 29 automated tests
py src/workload_generator.py # this will create the 5 workload files that will create the prediction error from 0-80%
py pipeline/generate_ood_workload.py # this will create the out-of-distribution workload files
py src/run_simulation.py # this will run the simulation for each scheduler (FCFS, LTR, Robust and LJF) in all 4 scenarios (light, normal, stress and saturation)
py pipeline/sweep.py # this will run all the combinations or the alpha and beta (total of 3240)
py pipeline/select_best_params.py # this will create the best parameters of alpha and beta per scenario and error levels
py pipeline/generate_tuned_robust.py  # this will rerun the  Robust scheduler with a tuned point for each scenario
py -c "import sys; sys.path.insert(0,'pipeline'); import sweep; sweep.run_ablation()" #This updates the ablation results
py pipeline/run_ood_test.py # this will run the out-of-distribution workloads for FCFS, LTR and Robust
```

3. The following part is not necessary for the simulation. They will create all 9 figures we included in the final written report

```bash
py scripts/fig1_jct_vs_error.py
py scripts/fig2_starvation_vs_error.py
py scripts/fig3_heatmap.py
py scripts/fig4_ablation.py
py scripts/fig5_pareto.py
py scripts/fig6_advantage_vs_fcfs.py
py scripts/fig7_jains_fairness.py
py scripts/fig8_ood_jct.py
py scripts/fig9_ood_starvation.py
```

4. The dashboard we created it`s a visual representation of the performance of our scheduler and a faster way to find the best arrangements for our formula in each scenario proposed. It works as and interactive demo where you can pick a load scenario, the level of prediction error and tune the alpha and beta values live for our scheduler (the FCFS and LTR dont have changeble parameters). It was also created a default configuration that shows the best Job COmpletion Time and Starvation percentage across all the error levels.


```bash
streamlit run dashboard/dashboard.py
```

## Shared request

In our simulator, every module reads and writes the object `RequestPacket` that is defined in the `src/shared_structures.py`. The fields are used like showed bellow:

**request_id** -> Is used by all modules as an unique request identifier

**arrival_time** -> Is used mainly by the `engine.py` and simulates the time that each request enters in the system to be answered

**actual_blocks** -> Is used by the `engine.py` and represents the real request size in KV-cache blocks and only the simulation engine can see

**predicted_mu** -> Is used mainly by the `schedulers.py` and is the prediction that the system does for the block cost of the request. The scheduler just sees this guess not the actual_blocks

**predicted_sigma** -> Is used by the `robust_scheduler.py` and measures how confident the system is about the prediction (The higher the sigma, the higher the uncertainty)

**wait_time** -> The `engine.py` updates this value every simulated instant, and it shows how long a request is waiting in line

**preemptions** -> Is used by the `metrics.py` and shows the risky predictions that the model made for a resquest (wrong predictions), that would mean a preemption in a real situation

**time_out** -> Is updated by the `engine.py` and represents the requests that were canceled because they waited too much. The maximum wait time was decided as 300s.



## Final results and report

This are the best configuration found for our scheduler in each of the scenarios:

**Light scenario** (Private LLM) -> alpha: 0.25, beta: 18, Average JCT: 76,5s, starvation: 1,5-2% (all errors levels), Advantage over FCFS in JCT: 9,7%

**Normal scenario** (Commercial Production Service) -> alpha: 0.25, beta: 18, Average JCT: 101,5s, starvation: 0-0,64% (all errors levels), Advantage over FCFS in JCT: 42,7%

**Stress scenario** (Peak demand) -> alpha: 0.25, beta: 18, Average JCT: 103s, starvation: 0-1,25% (all errors levels), Advantage over FCFS in JCT: 55,9%

**Saturation scenario** (Limit Capacity) -> alpha: 0.5, beta: 18, Average JCT: 105s, starvation: 0-1,66% (all errors levels), Advantage over FCFS in JCT: 58,1%