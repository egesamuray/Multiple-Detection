import os
import torch
import torch.distributed as dist

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass

def dev():
    if torch.cuda.is_available():
        lr = int(os.environ.get("LOCAL_RANK", 0))
        try: return torch.device(f"cuda:{lr}")
        except: return torch.device("cuda")
    return torch.device("cpu")

def using_distributed():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    if using_distributed():
        try: return dist.get_world_size()
        except: return 1
    return 1

def get_rank():
    if using_distributed():
        try: return dist.get_rank()
        except: return 0
    return 0

def is_master(): return get_rank() == 0

def synchronize():
    if using_distributed():
        try: dist.barrier()
        except: pass

def barrier(): synchronize()

@torch.no_grad()
def sync_params(params):
    if using_distributed():
        for p in params: dist.broadcast(p.data, src=0)

@torch.no_grad()
def all_reduce(tensor, op=dist.ReduceOp.SUM):
    if using_distributed(): dist.all_reduce(tensor, op=op)
    return tensor

def broadcast(obj, src=0): return obj
