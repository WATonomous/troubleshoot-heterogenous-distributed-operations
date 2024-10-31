#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1024  // Number of elements to send and receive

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        std::cerr << "This test requires at least two ranks." << std::endl;
        MPI_Finalize();
        return -1;
    }

    // Only rank 0 on the CUDA machine performs the send/recv
    if (rank == 0) {
        int* device_send_buffer;
        int* device_recv_buffer;
        int* host_send_buffer = new int[DATA_SIZE];
        int* host_recv_buffer = new int[DATA_SIZE];

        // Initialize the host send buffer with data
        for (int i = 0; i < DATA_SIZE; i++) {
            host_send_buffer[i] = i;
        }

        // Allocate device memory and copy data from host to device
        cudaMalloc((void**)&device_send_buffer, DATA_SIZE * sizeof(int));
        cudaMalloc((void**)&device_recv_buffer, DATA_SIZE * sizeof(int));
        cudaMemcpy(device_send_buffer, host_send_buffer, DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        // Send data to Rank 1 and receive data from Rank 1
        std::cout << "Rank 0 (CUDA) sending data...";
        MPI_Send(device_send_buffer, DATA_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Rank 0 (CUDA) sent data";
        MPI_Recv(device_recv_buffer, DATA_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy received data back to host for verification
        cudaMemcpy(host_recv_buffer, device_recv_buffer, DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

        // Print received data for verification
        std::cout << "Rank 0 (CUDA) received data: ";
        for (int i = 0; i < 5; i++) { // Print first few elements
            std::cout << host_recv_buffer[i] << " ";
        }
        std::cout << "..." << std::endl;

        // Cleanup
        delete[] host_send_buffer;
        delete[] host_recv_buffer;
        cudaFree(device_send_buffer);
        cudaFree(device_recv_buffer);
    }

    MPI_Finalize();
    return 0;
}
