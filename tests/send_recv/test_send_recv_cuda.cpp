#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>

#define BUFFER_SIZE 1024

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check that there are at least 2 ranks
    if (size < 2) {
        if (rank == 0) {
            std::cerr << "This program requires at least 2 MPI ranks." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Allocate device memory for the buffer
    float* device_buffer;
    cudaMalloc((void**)&device_buffer, BUFFER_SIZE * sizeof(float));

    if (rank == 0) {
        // Initialize the buffer on the host and copy it to the device
        float host_buffer[BUFFER_SIZE];
        for (int i = 0; i < BUFFER_SIZE; i++) {
            host_buffer[i] = static_cast<float>(i); // Fill the buffer with some values
        }

        // Copy the host buffer to the device
        cudaMemcpy(device_buffer, host_buffer, BUFFER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        // Send the buffer from rank 0 to rank 1
        MPI_Send(device_buffer, BUFFER_SIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        std::cerr << "Rank 0 sent data to Rank 1." << std::endl;
    } 
    else if (rank == 1) {
        // Receive the buffer from rank 0
        MPI_Recv(device_buffer, BUFFER_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cerr << "Rank 1 received data from Rank 0." << std::endl;

        // Copy the data back to the host for verification
        float host_buffer[BUFFER_SIZE];
        cudaMemcpy(host_buffer, device_buffer, BUFFER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        // Print the received data for verification
        std::cerr << "Received data: ";
        for (int i = 0; i < 10; i++) { // Print the first 10 elements for brevity
            std::cerr << host_buffer[i] << " ";
        }
        std::cerr << "..." << std::endl;
    }

    // Free device memory
    cudaFree(device_buffer);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
