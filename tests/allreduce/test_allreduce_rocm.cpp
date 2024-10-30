#include <mpi.h>
#include <hip/hip_runtime.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float *sendbuf, *recvbuf;
    size_t num_elements = 1024;

    // Allocate HIP memory and report memory type
    hipMalloc((void**)&sendbuf, num_elements * sizeof(float));
    hipMalloc((void**)&recvbuf, num_elements * sizeof(float));

    std::cout << "ROCm Rank " << rank << std::endl;

    // Initialize data
    float val = static_cast<float>(rank);
    hipMemcpy(sendbuf, &val, sizeof(float), hipMemcpyHostToDevice);

    // Perform MPI Allreduce
    MPI_Allreduce(sendbuf, recvbuf, num_elements, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    float result;
    hipMemcpy(&result, recvbuf, sizeof(float), hipMemcpyDeviceToHost);
    std::cout << "ROCm Rank " << rank << " received allreduce result: " << result << std::endl;

    hipFree(sendbuf);
    hipFree(recvbuf);

    MPI_Finalize();
    return 0;
}
