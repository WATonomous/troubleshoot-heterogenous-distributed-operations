import os
import torch
import torch.distributed as dist

# Environment variables set by the MPI launch system
LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])


def run_allreduce():
    # Initialize tensor with a unique value per rank
    tensor = torch.ones(1) * WORLD_RANK

    # Move tensor to appropriate GPU
    device = torch.device(f"cuda:{LOCAL_RANK}")
    tensor = tensor.to(device)

    print(f"Before all-reduce, Rank {WORLD_RANK} has tensor: {tensor.item()}")

    # Perform all-reduce operation on GPU tensor, summing values across all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"After all-reduce, Rank {WORLD_RANK} has tensor: {tensor.item()}")


def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run_allreduce()


if __name__ == "__main__":
    init_processes(backend="mpi")
