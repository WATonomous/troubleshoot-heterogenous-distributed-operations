#include <iostream>
#include <mpi.h>
#include <hip/hip_runtime.h>

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
        hipMalloc((void**)&device_send_buffer, DATA_SIZE * sizeof(int));
        hipMalloc((void**)&device_recv_buffer, DATA_SIZE * sizeof(int));
        hipMemcpy(device_send_buffer, host_send_buffer, DATA_SIZE * sizeof(int), hipMemcpyHostToDevice);

        // Send data to Rank 1 and receive data from Rank 1
        std::cerr << "Rank 0 (ROCm) sending data...";
        MPI_Send(device_send_buffer, DATA_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cerr << "Rank 0 (ROCm) sent data";
        MPI_Recv(device_recv_buffer, DATA_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy received data back to host for verification
        hipMemcpy(host_recv_buffer, device_recv_buffer, DATA_SIZE * sizeof(int), hipMemcpyDeviceToHost);

        // Print received data for verification
        std::cerr << "Rank 0 (ROCm) received data: ";
        for (int i = 0; i < 5; i++) { // Print first few elements
            std::cerr << host_recv_buffer[i] << " ";
        }
        std::cerr << "..." << std::endl;

        // Cleanup
        delete[] host_send_buffer;
        delete[] host_recv_buffer;
        hipFree(device_send_buffer);
        hipFree(device_recv_buffer);
    } else if (rank == 1) {
        int* device_send_buffer;
        int* device_recv_buffer;
        int* host_send_buffer = new int[DATA_SIZE];
        int* host_recv_buffer = new int[DATA_SIZE];

        // Initialize the host send buffer with data
        for (int i = 0; i < DATA_SIZE; i++) {
            host_send_buffer[i] = i + 1000;
        }

        // Allocate device memory and copy data from host to device
        hipMalloc((void**)&device_send_buffer, DATA_SIZE * sizeof(int));
        hipMalloc((void**)&device_recv_buffer, DATA_SIZE * sizeof(int));
        hipMemcpy(device_send_buffer, host_send_buffer, DATA_SIZE * sizeof(int), hipMemcpyHostToDevice);

        // Receive data from Rank 0
        MPI_Recv(device_recv_buffer, DATA_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy received data back to host for verification
        hipMemcpy(host_recv_buffer, device_recv_buffer, DATA_SIZE * sizeof(int), hipMemcpyDeviceToHost);

        // Print received data for verification
        std::cerr << "Rank 1 (ROCm) received data: ";
        for (int i = 0; i < 5; i++) { // Print first few elements
            std::cerr << host_recv_buffer[i] << " ";
        }
        std::cerr << "..." << std::endl;

        // Send data to rank 1
        MPI_Send(device_send_buffer, DATA_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
        std::cerr << "Rank 1 (ROCm) sent data";

        // Cleanup
        delete[] host_send_buffer;
        delete[] host_recv_buffer;
        hipFree(device_send_buffer);
        hipFree(device_recv_buffer);
    }

    MPI_Finalize();
    return 0;
}
