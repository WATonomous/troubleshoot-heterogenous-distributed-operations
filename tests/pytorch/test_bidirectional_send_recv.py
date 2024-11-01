import os
import torch
import torch.distributed as dist

# Environment variables set by the MPI launch system
LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])


def run_bi_directional_send_recv():
    # Initialize a tensor with a unique value per rank and move it to the appropriate GPU
    device = torch.device(f"cuda:{LOCAL_RANK}")
    send_tensor = torch.tensor([WORLD_RANK], dtype=torch.float32).to(device)
    recv_tensor = torch.zeros(1, dtype=torch.float32).to(device)

    # Define partner rank (even ranks communicate with next odd rank and vice versa)
    if WORLD_RANK % 2 == 0:
        partner_rank = WORLD_RANK + 1
    else:
        partner_rank = WORLD_RANK - 1

    # Ensure partner rank is within the world size
    if partner_rank < WORLD_SIZE:
        if WORLD_RANK % 2 == 0:
            # Even ranks send first, then receive
            dist.send(tensor=send_tensor, dst=partner_rank)
            print(
                f"Rank {WORLD_RANK} sent data to Rank {partner_rank}: {send_tensor.item()} on device {device}"
            )

            dist.recv(tensor=recv_tensor, src=partner_rank)
            print(
                f"Rank {WORLD_RANK} received data from Rank {partner_rank}: {recv_tensor.item()} on device {device}"
            )
        else:
            # Odd ranks receive first, then send
            dist.recv(tensor=recv_tensor, src=partner_rank)
            print(
                f"Rank {WORLD_RANK} received data from Rank {partner_rank}: {recv_tensor.item()} on device {device}"
            )

            dist.send(tensor=send_tensor, dst=partner_rank)
            print(
                f"Rank {WORLD_RANK} sent data to Rank {partner_rank}: {send_tensor.item()} on device {device}"
            )


def init_processes():
    # Initialize the process group with 'mpi' backend
    dist.init_process_group(backend="mpi", rank=WORLD_RANK, world_size=WORLD_SIZE)
    run_bi_directional_send_recv()


if __name__ == "__main__":
    init_processes()
