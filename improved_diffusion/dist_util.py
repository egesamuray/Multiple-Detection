# improved_diffusion/dist_util.py
import os
import torch
import torch.distributed as dist

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INITIALIZED = False

def setup_dist():
    """Best-effort init. If env vars absent, run single-proc/single-GPU."""
    global _DEVICE, _INITIALIZED
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
            _DEVICE = torch.device(f"cuda:{local_rank}")
        except Exception:
            _DEVICE = torch.device("cuda")
    else:
        _DEVICE = torch.device("cpu")

    if not dist.is_available() or dist.is_initialized():
        _INITIALIZED = True
        return

    world_size = os.environ.get("WORLD_SIZE")
    rank = os.environ.get("RANK")
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")

    if world_size and rank and master_addr and master_port:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except Exception:
            pass
    _INITIALIZED = True

def dev():
    return _DEVICE

def using_distributed():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    if using_distributed():
        try: return dist.get_world_size()
        except Exception: return 1
    return 1

def get_rank():
    if using_distributed():
        try: return dist.get_rank()
        except Exception: return 0
    return 0

def is_master():
    return get_rank() == 0

def synchronize():
    if using_distributed():
        try: dist.barrier()
        except Exception: pass

# alias so legacy code calling dist.barrier() still works
def barrier():
    synchronize()

@torch.no_grad()
def sync_params(params):
    if using_distributed():
        for p in params:
            dist.broadcast(p.data, src=0)

@torch.no_grad()
def all_reduce(tensor, op=dist.ReduceOp.SUM):
    if using_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor

def broadcast(obj, src=0):
    return obj
