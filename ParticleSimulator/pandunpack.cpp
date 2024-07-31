#include<mpi.h>
#include<iostream>
#include<vector> 



int main(int argc, char** argv) {
    int rank, size;
    std::vector<int> vec1 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> vec2 {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};

    MPI_Status status;
    char buffer[100];
    int position = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank ==0) {
        MPI_Pack(&vec1[0], 2, MPI_INT, buffer, 100, &position, comm);
        MPI_Pack(&vec1[5], 1, MPI_INT, buffer, 100, &position, comm);
        MPI_Send(buffer, 100, MPI_PACKED, 1, 0, comm);
    }

    if (rank==1) {

        MPI_Recv(buffer, 100, MPI_PACKED, 0, 0, comm, &status);
        MPI_Unpack(buffer, 100, &position, &vec2[1], 2, MPI_INT, comm);
        MPI_Unpack(buffer, 100, &position, &vec2[8], 1, MPI_INT, comm);

        // for (int i = 0; i < 10; i++) {
        //     std::cout << vec2[i] << " ";
        // }
        for (auto &v: vec2) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
    std::cout  << "DONE:"<< std::endl;
    MPI_Finalize();
    return 0;

}