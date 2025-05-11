# Troubleshooting Collective Operations on a Heterogeneous Cluster with UCC and UCX

This repo is a proof-of-concept setup to run workloads across ROCm and CUDA. This is a fork of the original (excellent) work by Rafał Siwek (https://github.com/RafalSiwek/troubleshoot-heterogenous-distributed-operations) with some bug fixes and additional experiments.

## Getting started

Start the containers:

```sh
docker compose up --build
```

Extract the SSH keys from the containers:

```sh
(docker compose exec rocm1 cat /root/.ssh/id_ed25519.pub && docker compose exec cuda1 cat /root/.ssh/id_ed25519.pub) > tmp/ssh-pub-keys.txt
```

Copy the public keys to each worker container by pasting it into the `/root/.ssh/authorized_keys` file:

```sh
cat tmp/ssh-pub-keys.txt | docker compose exec --no-TTY rocm1 tee /root/.ssh/authorized_keys
cat tmp/ssh-pub-keys.txt | docker compose exec --no-TTY cuda1 tee /root/.ssh/authorized_keys
```

Check that each machine can ssh into the other:

```sh
# should print "cuda1"
docker compose exec rocm1 ssh root@cuda1 hostname

# should print "rocm1"
docker compose exec cuda1 ssh root@rocm1 hostname
```

## Run tests

### Example test (`send_recv`)

Compile the test:

```sh
docker compose exec rocm1 bash

# in the container
cp -r tests/send_recv /tmp/
cd /tmp/send_recv
hipcc test_send_recv_rocm.cpp -lmpi
```

```sh
docker compose exec cuda1 bash

# in the container
cp -r tests/send_recv /tmp/
cd /tmp/send_recv
nvcc test_send_recv_cuda.cpp -lmpi
```

Run the test (in one of the containers):

```sh
docker compose exec rocm1 bash

mpirun --allow-run-as-root -np 2 -H rocm1,cuda1 \
-mca pml ucx -mca coll_ucc_enable 1 -mca coll_ucc_priority 100 \
-mca coll_ucc_verbose 3 -mca pml_ucx_verbose 3 \
/tmp/send_recv/a.out

# make sure the command exited successfully. Should print "0"
echo $?
```

# Old README

This repository provides a proof-of-concept setup to run HPC workloads on AWS public cloud instances with different GPU accelerators. It also documents the issues encountered during deployment.

## Infrastructure Setup

This workload is configured for two types of [g4](https://aws.amazon.com/ec2/instance-types/g4/) instances, each with distinct GPU types:

- **g4ad.xlarge**: One AMD Radeon Pro V520 GPU (gfx1011) using [ROCm 6.2.2](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).
- **g4dn.xlarge**: One NVIDIA T4 GPU using [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local).

These instances are running Ubuntu 22.04 (`ubuntu-eks/k8s_1.30/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*` AMI images). Since g4ad instances don’t support [EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html), networking was configured for standard performance with bandwidth up to 10 Gbps and security groups allowing traffic on all ports.

## Collective Operations Environment

Collective operations run as distributed PyTorch jobs in OCI containers.

Separate Docker images were built for each instance type and published to Docker Hub:

- **[g4ad](./g4ad_rocm_build/Dockerfile)**: Required building ROCm algebraic libraries from source to support the gfx1011 architecture.
    - [UCX configuration](./config_infos/rocm_ucx_config)
    - [UCC configuration](./config_infos/rocm_ucc_config)

- **[g4dn](./g4dn_cuda_build/Dockerfile)**: Built with compatible versions of collective libraries (OMPI, UCX, UCC).
    - [UCX configuration](./config_infos/cuda_ucx_config)
    - [UCC configuration](./config_infos/cuda_ucc_config)

Additional tests were conducted using [PyTorch distributed backends](https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends) for the `torchrun` setup, but with no successful outcomes:
- [nccl <--> rccl](https://github.com/ROCm/rccl/issues/1220)
- [ucc](https://github.com/openucx/ucx/discussions/9985)

Following guidance from the UCX team ([response here](https://github.com/openucx/ucx/discussions/9985)), we switched to using MPI with UCC and UCX.

## Running Collective Operations

To test collective operations, start three containers using the Docker images with host network configurations (pre-installed PyTorch images are available):

- **g4ad**:
    ```bash
    docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    --shm-size 8G --user root --rm --network host \
    rafalsiwek/g4ad_distributed_ml:1.0_pytorch bash
    ```

- **g4dn**:
    ```bash
    docker run --gpus all -it --rm --user root --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined --ipc=host --shm-size 8G \
    --network host --group-add video \
    rafalsiwek/g4dn_distributed_ml:1.0_pytorch bash
    ```

- **t3**
    ```bash
    docker run -it --rm --network host rafalsiwek/opmpi_ucx_simple:latest
    ```

These MPI worker containers are set up for passwordless SSH. Follow these steps:

1. In the "master container," generate an SSH key:
    ```bash
    ssh-keygen -t rsa
    ```

2. Copy the public key to each "worker container" by pasting it into the `~/.ssh/authorized_keys` file.

3. Update the SSH daemon (`sshd`) port in each worker container to a port not used by the host:
    ```bash
    vi /etc/ssh/sshd_config
    ```

4. Change the SSH port in the "master container":
    ```bash
    vi /etc/ssh/ssh_config
    ```

5. Start the SSH server in each worker container:
    ```bash
    /usr/sbin/sshd -D
    ```

## Tests Run

The [tests](./tests/) directory contains scripts and log outputs for the tests conducted.

### [Send_Recv](./tests/send_recv/)

This test sends and receives a GPU memory buffer between nodes. The CUDA node generated a buffer with `cudaMemcpy`, while the ROCm node used `hipMemcpy`.

After compiling with `nvcc` on the CUDA node and `hipcc` on the ROCm node, the MPI job was triggered with:
```bash
mpirun --allow-run-as-root -np 2 -H <rocm_ip>,<cuda_ip> \
-mca pml ucx -mca coll_ucc_enable 1 -mca coll_ucc_priority 100 \
-mca coll_ucc_verbose 3 -mca pml_ucx_verbose 3 \
/test_send_recv
```
This job completed successfully ([log output](./tests/send_recv/test_run_send_recv_mpi.log)). The same job was also run with `-x UCC_LOG_LEVEL=DEBUG` ([log output](./tests/send_recv/test_run_send_recv_mpi_ucc_verbose.log)).

### [Bidirectional Send_Recv](./tests/bidirectional_send_recv/)

This test runs a bidirectional simple send and receive operation where Rank 0 (CUDA) sends and recvs data from Rank1 (ROCM)

The job was triggered wuth:
```bash
mpirun --allow-run-as-root -np 1 -host <cuda_ip> \
-mca pml ucx -mca coll_ucc_enable 1 -mca coll_ucc_priority 100 \
-x UCC_LOG_LEVEL=DEBUG -x UCC_COLL_TRACE=DEBUG \
/test_bidirectional_send_recv : \
-np 1 -host <rocm_ip> \
-mca pml ucx -mca coll_ucc_enable 1 -mca coll_ucc_priority 100 \
-x UCC_LOG_LEVEL=DEBUG -x UCC_COLL_TRACE=DEBUG \
/test_bidirectional_send_recv
```
An error occurred on the ROCm node/rank with the `ucp_mem_type_unpack` method ([log output with backtrace](./tests/bidirectional_send_recv/test_run_bidirectional_send_recv_mpi_cuda_to_rocm_ucc_debug_dump.log)).

### [AllReduce](./tests/allreduce/)

This test involves running an allreduce operation in a heterogeneous ring. As in the `send_recv` test, the CUDA node generated a buffer with `cudaMemcpy`, while the ROCm node used `hipMemcpy`.

The job was run with:
```bash
mpirun --allow-run-as-root -np 2 -H <rocm_ip>,<cuda_ip> \
-mca pml ucx -mca coll_ucc_enable 1 -mca coll_ucc_priority 100 \
-mca coll_ucc_verbose 3 -mca pml_ucx_verbose 3 \
/test_allreduce
```
An error occurred on the ROCm node/rank with the `ucp_mem_type_unpack` method ([log output with backtrace](./tests/allreduce/test_run_allreduce_mpi_basic.log)).

Suspecting UCX was using different TLs, the job was re-run with the `-mca pml_ucx_tls=tcp` option to force TCP and `-mca pml_ucx_devices ens4` to specify the `ens4` network device. However, the failure persisted ([log output](./tests/allreduce/test_run_allreduce_mpi_selected_ucx_tl.log)).

Following recommendations in [UCC Issue #1039](https://github.com/openucx/ucc/issues/1039), the job was run with `-x UCC_TL_UCP_TUNE=inf` to adjust the UCP transport layer for UCC. ***Although the ROCm node/rank failed, the CUDA node/rank completed the allreduce job ([log output](./tests/allreduce/test_run_allreduce_mpi_tuned_ucp.log)).***

Still suspecting integration issues between UCC and UCX, debug logging for UCX was enabled with `-x UCX_LOG_LEVEL=DEBUG` ([log output](./tests/allreduce/test_run_allreduce_mpi_tuned_ucx_debug_dump.log)). The `send_recv` example was also re-run with UCX debug logs ([log output](./tests/send_recv/test_run_send_recv_mpi_tuned_ucx_debug_dump.log)), though the logs did not provide clear conclusions for me.

Additional tests were run for the `allreduce` collective on a homogeneous setup with CUDA-only and ROCm-only environments. The CUDA-only ring tests were successful ([UCC debug and trace logs here](./tests/allreduce/test_run_allreduce_cuda_only_ucc_debug_dump.log)). However, the ROCm-only ring encountered errors with the `uct_rocm_copy_ep_put_short` function in the `ucx/libuct_rocm` library, which is consistent with the errors seen in previous tests.

To gather more details, I ran additional jobs to capture UCX debug logs ([UCX logs with `-x UCX_LOG_LEVEL=DEBUG` here](./tests/allreduce/test_run_allreduce_rocm_only_ucx_debug_dump.log) and [logs with `-mca pml_ucx_verbose 100` here](./tests/allreduce/test_run_allreduce_rocm_only_ucx_pml_debug_dump.log)).

To verify that the issue is specific to ROCm-only ring communication, I also tested simple `send_recv` operations. These showed the same error ([logs with `-x UCX_LOG_LEVEL=DEBUG` here](./tests/send_recv/test_run_send_recv_rocm_only_ucx_debug_dump.log) and [logs with `-mca pml_ucx_verbose 100` here](./tests/send_recv/test_run_send_recv_rocm_only_ucx_pml_debug_dump.log)). In contrast, the `send_recv` test on the CUDA-only ring completed without issues.

As a result the root of the `uct_rocm_copy_ep_put_short` issue with ROCm ranks was related to the fact that AWS EC2 g4ad machines do not supprt (Large BAR setting)[https://github.com/openucx/ucx/wiki/Build-and-run-ROCM-UCX-OpenMPI#sanity-check-for-large-bar-setting] and can be circumvent with the following variables (for more context look [here](https://github.com/openucx/ucc/issues/1043)):
```bash
UCX_ROCM_COPY_D2H_THRESH=0
UCX_ROCM_COPY_H2D_THRESH=0
UCC_EC_ROCM_REDUCE_HOST_LIMIT=0
UCC_EC_ROCM_COPY_HOST_LIMIT=0
OMPI_MCA_mpi_accelerator_rocm_memcpyD2H_limit=0
OMPI_MCA_mpi_accelerator_rocm_memcpyH2D_limit=0
```


### [PyTorch](./tests/pytorch/)

To test ML workflows, I built PyTorch (v2.5.1) from source with MPI support following [this guide](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/build.sh). Experiments were run using `mpirun` with UCX PML and UCC collective configurations.

Running a [bi-directional send/recv test](./tests/pytorch/test_bidirectional_send_recv.py) was successful, confirming basic communication across GPUs ([logs available here](./tests/pytorch/test_run_bidirectional_send_recv_mpi_ucc.log)).

However, testing collective operations with UCC, such as the [allreduce test](./tests/pytorch/test_allreduce.py), and also with the official [PyTorch distributed examples](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py), led to a failure on the `ucp_tag_send_nbx` operation for both ranks ([logs and backtrace available here](./tests/pytorch/test_run_allreduce_mpi_with_ucc_debug.log)).

Running the allreduce test with UCC collectives disabled for MPI yielded partial success, where the operation completed successfully on the CUDA rank but failed on the ROCm rank ([logs and backtrace here](./tests/pytorch/test_run_allreduce_mpi_only.log)).

The appeared to be related to the UCX configuration as the worker fails at `ucs_async_check_owner_thread(&(ep->worker)->async)` assertion. UCX at configuration time had enabled multi-threading support with `--enable-mt` flag, which this setup might not fully support. After rebuilding UCX with multi-threading disabled, everything seems to be working correctly.


### [OSU Micro-Benchmarks](./tests/osu_benchmarks/)

To verify the UCX portion of the software stack, [OSU Micro-Benchmarks 7.4](https://mvapich.cse.ohio-state.edu/benchmarks/) were conducted. Separate builds of the benchmark were created for each type of rank to match the specific GPU hardware: one with CUDA for NVIDIA and one with ROCm for AMD. The benchmarks were run using the GPU device, and here are the test results:

- **P2P Bi-Directional Bandwidth Benchmark** ([results and logs here](./tests/osu_benchmarks/osu_bibw.log)) - This test completed successfully for all data types available, validating both the benchmark and the setup. The results reflect the bandwidth limitations of the setup.

- **Collective AllGather Latency Benchmark** ([results and logs here](./tests/osu_benchmarks/osu_allgather.log)) - The first collective test ran successfully across all available data types, showing expected operation latencies without any errors.

- **Collective AllReduce Latency Benchmark** - This is the primary and most extensive collective test. Running the benchmark with validation disabled was successful ([results and logs here](./tests/osu_benchmarks/osu_allreduce.log)). However, enabling validation led to failures across all tested data types, with logs showing significant discrepancies between expected and actual values ([results and logs here](./tests/osu_benchmarks/osu_allreduce_validation.log)). It is worth noting that, since version 7.2, the OSU benchmark suite updated its validation methods for reduction tests ([see changelog](https://mvapich.cse.ohio-state.edu/static/media/mvapich/CHANGES-OMB.txt)). When re-running the benchmark with version 7.2, validation passed without issues ([results and logs here](./tests/osu_benchmarks/osu_7_2_allreduce_validation.log)).

After these tests, it appears that this heterogeneous setup can handle collective communication with some inherent bandwidth and latency limitations. This raises the question of why collective operations in PyTorch fail on the `ucp_tag_send_nbx` operation, while basic bi-directional `send_recv` operations complete without issue. Could this be due to the way PyTorch implements collective operations? It seems possible, even though PyTorch uses [similar functions](https://github.com/pytorch/pytorch/blob/a979318ef7fabac7e0d7a2101e0e70af75fca7bd/torch/csrc/distributed/c10d/ProcessGroupMPI.cpp#L432) to those in [my allreduce test](./tests/allreduce/test_allreduce_rocm.cpp).

## Solution

This excercise allowed to implement two main solutions to the problem:

- To enable collective operations with a machine not supporting Large BAR, in this case the ROCm Rank node with AMD ROCm, following variables have to be passed with `mpirun`:
    ```bash
    UCX_ROCM_COPY_D2H_THRESH=0
    UCX_ROCM_COPY_H2D_THRESH=0
    UCC_EC_ROCM_REDUCE_HOST_LIMIT=0
    UCC_EC_ROCM_COPY_HOST_LIMIT=0
    OMPI_MCA_mpi_accelerator_rocm_memcpyD2H_limit=0
    OMPI_MCA_mpi_accelerator_rocm_memcpyH2D_limit=0
    ```
- To avoid failing PyTorch jobs due to possible lack of UCX multi-thread support, UCX has to be rebuilt without the `--enable-mt` flag.

Thanks for help to:
 - [edgargabriel](https://github.com/edgargabriel)
 - [Sergei-Lebedev](https://github.com/Sergei-Lebedev)