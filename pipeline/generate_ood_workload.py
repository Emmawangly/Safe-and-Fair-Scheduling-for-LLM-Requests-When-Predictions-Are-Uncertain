import random
import math
import json

NUM_REQUESTS = 1000
OOD_LEVELS = [0, 20, 40, 60, 80]
SEED = 42


def sample_actual_blocks(rng):
    raw = rng.lognormvariate(1.5, 1.0)
    return max(1, min(200, int(raw)))


def create_request(request_id, rng, arrival_time, ood_fraction):
    actual_blocks = sample_actual_blocks(rng)
    is_ood = rng.random() < ood_fraction

    if is_ood:
        # fixed thw sigma!
        predicted_mu = max(1.0, sample_actual_blocks(rng))
        predicted_sigma = predicted_mu * 0.5 / math.sqrt(3)
    else:
        
        noise_factor = rng.uniform(0.9, 1.1)
        predicted_mu = max(1.0, actual_blocks * noise_factor)
        predicted_sigma = predicted_mu * 0.1 / math.sqrt(3)

    return {
        "request_id": request_id,
        "arrival_time": round(arrival_time, 3),
        "actual_blocks": actual_blocks,
        "predicted_mu": round(predicted_mu, 3),
        "predicted_sigma": round(predicted_sigma, 3),
        "wait_time": 0.0,
    }


def generate_workload(num_requests, ood_fraction, seed=SEED):
    rng = random.Random(seed)
    queue = []
    current_time = 0.0

    for request_id in range(1, num_requests + 1):
        current_time += rng.expovariate(1.0)
        queue.append(create_request(request_id, rng, current_time, ood_fraction))

    return queue


def save_queue_as_json(queue, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(queue, file, indent=2)


def main():
    for ood_pct in OOD_LEVELS:
        queue = generate_workload(NUM_REQUESTS, ood_pct / 100)
        output_path = f"data/workload_ood_{ood_pct}.json"
        save_queue_as_json(queue, output_path)
        print(f"saved {len(queue)} requests with {ood_pct}% out-of-distribution to {output_path}")


if __name__ == "__main__":
    main()
