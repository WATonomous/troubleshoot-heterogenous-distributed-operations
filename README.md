# Troubles running collective operations on heterogenous cluster with UCC and UCX

This repository outlines the approach for a proof of concept that runs HPC workloads on heterogeneous AWS public cloud instances with varying GPU accelerators, along with the issues encountered during execution.

## Infrastructure setup

The workload was run on two types of [g4](https://aws.amazon.com/ec2/instance-types/g4/) instances with varying GPU acceleration nodes:

- **g4ad.xlarge** with one AMD Radeon Pro V520 GPU (gfx1011) using the [ROCm 6.2.2](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) toolkit
- **g4dn.xlarge** with one NVIDIA T4 GPU using the [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) toolkit

The instances were running the Ubuntu 22.04 OS (`ubuntu-eks/k8s_1.30/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*`-based AMI images).

Since g4ad instances do not support [EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html), networking was configured using the standard setup, with bandwidth up to 10 Gigabit and security groups allowing traffic on all ports

## Collectiove operations environment setup

The collective operations are intended to run within OCI containers as distributed PyTorch jobs.

For both instance types, dedicated images were built and published to my dockerhub:
- **[g4ad](./g4ad_rocm_build/Dockerfile)** required building most of the ROCm algebraic libraries from source to support the gfx1011 architecture.
- **[g4dn](./g4dn_cuda_build/Dockerfile)** was built to maintain consistency across collective libraries (OMPI, UCX, UCC).


In addition to running the workload, these images were also used to build PyTorch (`v2.5.0-rc7`) from source with NCCL, MPI, and UCC support, following [this guide](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/build.sh).

Several [backends](https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends) for the `torchrun` approach were tested, but without success:
- [nccl <--> rccl](https://github.com/ROCm/rccl/issues/1220)
- [ucc](https://github.com/openucx/ucx/discussions/9985)

Suspecting potential issues related to limited networking configuration and encouraged by the [response](https://github.com/openucx/ucx/discussions/9985), I decided to migrate to using MPI with UCC and UCX support.

## Running collective operations

To run a simple PoC for collective operations configuration, I ran 3 containers using published Docker images with host network configuration (you can also use the images with PyTorch pre-installed):

- **g4ad** (there's also an image with UCX and UCC built from the ROCm fork source available with the tag `rocmucx`):
    ```bash
    docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    --shm-size 8G --user root --rm --network host \
    rafalsiwek/g4ad_pytorch_build:openucx bash
    ```

- **g4dn**:
    ```bash
    docker run --gpus all -it --rm --user root --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined --ipc=host --shm-size 8G \
    --network host --group-add video \
    rafalsiwek/g4dn_pytorch_build:latest bash
    ```

I decided to run MPI worker containers with passwordless SSH. To configure this, follow these steps:

1. In the "master container", generate an SSH key:
    ```bash
    ssh-keygen -t rsa
    ```

2. Copy the public key to the "worker containers" by pasting it into the `~/.ssh/authorized_keys` file.

3. Modify the SSH daemon (sshd) port on the worker containers to a port not used by the host instances:
    ```bash
    vi /etc/ssh/sshd_config
    ```

4. Change the default SSH port on the "master container":
    ```bash
    vi /etc/ssh/ssh_config
    ```

5. Run the SSH server on the worker containers:
    ```bash
    /usr/sbin/sshd -D
    ```

I began testing with the [full PyTorch distributed example](./scripts/test_full_train.py), using only the TCP transport layer:

```
mpirun -np 2 --allow-run-as-root --host <rocm_ip>,<cuda-ip> -x MASTER_ADDR=<cuda-ip> -x MASTER_PORT=1234 -x UCX_NET_DEVICES=ens5 --mca pml ucx --mca osc ucx --mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x TORCH_BLAS_PREFER_CUBLASLT=0 -x TORCH_BLAS_PREFER_HIPBLASLT=0 -x UCX_TLS=tcp -mca btl ^uct --mca pml_ucx_tls tcp /opt/conda/envs/py_3.12/bin/python /test_full_train.py 10 10
```
However, the test failed with the following error: 
```
Segmentation fault: invalid permissions for mapped object at address
```

The same happened when I tried to run a [simple test with just `send` and `recv` functions](./scripts/test_simple.py):

```
mpirun -np 2 --allow-run-as-root --host <rocm_ip>,<cuda-ip> -x MASTER_ADDR=<cuda-ip> -x MASTER_PORT=1234 -x UCX_NET_DEVICES=ens5 --mca pml ucx --mca osc ucx --mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x TORCH_BLAS_PREFER_CUBLASLT=0 -x TORCH_BLAS_PREFER_HIPBLASLT=0 -x UCX_TLS=tcp -mca btl ^uct --mca pml_ucx_tls tcp /opt/conda/envs/py_3.12/bin/python /test_simple.py
```
This resulted in the same segmentation fault.

In the [test_results](./test_results/) directory, I appended log files from different verbosity configurations passed to the mpirun command:

- ucc_verbose: with -x UCC_LOG_LEVEL=DEBUG
- ucx_verbose: with -x UCX_LOG_LEVEL=DEBUG
- ucx_pml_verbose: with --mca pml_ucx_verbose 100

The only successful result occurred in a simple scenario when running mpirun without UCX configuration:

```
mpirun -np 2 --allow-run-as-root --host <rocm_ip>,<cuda-ip> -x MASTER_ADDR=<cuda-ip> -x MASTER_PORT=1234 --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens5 /opt/conda/envs/py_3.12/bin/python /test_simple.py
```