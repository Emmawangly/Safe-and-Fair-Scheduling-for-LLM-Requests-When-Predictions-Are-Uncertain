from shared_structures import RequestPacket
#from schedulers import fcfs_scheduler
from schedulers import robust_scheduler
import csv

class SimulationEngine:
    def __init__(self, requests, time_step=1.0, blocks_per_second=4):
        
        
        self.all_requests = sorted(requests, key=lambda r: r.arrival_time)
        self.time_step = time_step
        self.current_time = 0.0

        self.queue = []        
        self.running = None      
        self.completed = []      

        
        self.used_blocks = 0
        self.blocks_per_second = blocks_per_second

       
        self.remaining_blocks = 0

        
        self.records = []

    def run(self, scheduler_fn=robust_scheduler, max_time=100.0, alpha=1.0, beta=1.0):
        while self.current_time <= max_time and (self.all_requests or self.queue or self.running):
            
            self._enqueue_arrivals()

            
            for req in self.queue:
                req.wait_time = self.current_time - req.arrival_time

            
            if self.running is None and self.queue:
                idx = scheduler_fn(self.queue, alpha=alpha, beta=beta)
                if idx is not None:
                    next_req = self.queue.pop(idx)
                    self._start_request(next_req)

            
            self._advance_running()
            self.current_time += self.time_step
    
    def _enqueue_arrivals(self):
        
        to_move = []
        for req in self.all_requests:
            if req.arrival_time <= self.current_time:
                to_move.append(req)
        # take from all_requests and send to queue
        for req in to_move:
            self.all_requests.remove(req)
            self.queue.append(req)

    def _start_request(self, req):
        req.ttft = req.wait_time
        self.running = req
        self.used_blocks += req.actual_blocks
        self.remaining_blocks = req.actual_blocks

        if req.actual_blocks > req.predicted_mu:
            req.preemptions += 1

    def _advance_running(self):
        
        if self.running is None:
            return

       
        self.remaining_blocks -= self.blocks_per_second * self.time_step

        if self.remaining_blocks <= 0:
            finished = self.running
            self.running = None
            self.used_blocks -= finished.actual_blocks
            self.completed.append(finished)

            self.records.append({
                "request_id": finished.request_id,
                "arrival_time": finished.arrival_time,
                "completion_time": self.current_time,
                "wait_time": finished.wait_time,
                "ttft": finished.ttft,
                "preemptions": finished.preemptions,
                "actual_blocks": finished.actual_blocks,
                "predicted_mu": finished.predicted_mu,
                "predicted_sigma": finished.predicted_sigma,
            })

    def write_csv(self, filename):
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    


