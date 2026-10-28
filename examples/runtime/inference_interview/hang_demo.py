"""Bounded CPU/Gloo P2P demonstration; no GPU or production job touched.

Run with: torchrun --standalone --nproc-per-node=2 hang_demo.py --mode bad
The bad mode intentionally times out. Good mode should print 'done' twice.
Use --pause 15 to leave time to inspect both worker stacks externally.
"""

import argparse
from datetime import timedelta
import os
import time
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("bad", "good"), default="good")
parser.add_argument("--pause", type=float, default=0)
args = parser.parse_args()
dist.init_process_group("gloo", timeout=timedelta(seconds=8))
if dist.get_world_size() != 2:
    dist.destroy_process_group()
    raise ValueError("This demonstration requires exactly two processes.")
rank = dist.get_rank()
peer = 1 - rank
tensor = torch.ones(1)
print(f"rank={rank} pid={os.getpid()} mode={args.mode}", flush=True)
time.sleep(args.pause)
try:
    if args.mode == "bad" or rank == 0:
        print(f"rank={rank} send", flush=True)
        dist.send(tensor, peer)
        dist.recv(tensor, peer)
    else:
        print(f"rank={rank} recv", flush=True)
        dist.recv(tensor, peer)
        dist.send(tensor, peer)
    print(f"rank={rank} done", flush=True)
finally:
    dist.destroy_process_group()
