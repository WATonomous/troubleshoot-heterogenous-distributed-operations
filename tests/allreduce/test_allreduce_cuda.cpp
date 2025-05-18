#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float *sendbuf, *recvbuf;
    size_t num_elements = 1024;

    // Allocate CUDA memory and report memory type
    cudaMalloc((void**)&sendbuf, num_elements * sizeof(float));
    cudaMalloc((void**)&recvbuf, num_elements * sizeof(float));

    std::cerr << "CUDA Rank " << rank << std::endl;

    // Initialize data
    float val = static_cast<float>(rank);
    cudaMemcpy(sendbuf, &val, sizeof(float), cudaMemcpyHostToDevice);

    // Perform MPI Allreduce
    MPI_Allreduce(sendbuf, recvbuf, num_elements, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    float result;
    cudaMemcpy(&result, recvbuf, sizeof(float), cudaMemcpyDeviceToHost);
    std::cerr << "CUDA Rank " << rank << " received allreduce result: " << result << std::endl;

    cudaFree(sendbuf);
    cudaFree(recvbuf);

    MPI_Finalize();
    return 0;
}
