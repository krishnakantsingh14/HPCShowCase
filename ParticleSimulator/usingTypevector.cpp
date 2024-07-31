#include<mpi.h>
#include<iostream>
#include<vector>

int main(int argc, char *argv[]) {

    int rank, size;
    int wvector[3][3] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request request;
    MPI_Status status;
    MPI_Datatype column_type;
    MPI_Type_vector(3, 1, 3, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);

    if (rank ==0) {
        
        
        int buffer[3][3] = {
            {1,2,3},
            {4,5,6},
            {7,8,9}
        };

        
        MPI_Send(&buffer[0][1], 1, column_type, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {

        

        MPI_Recv(&wvector[0][1], 1, column_type, 0, 0, MPI_COMM_WORLD, &status);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                std::cout << wvector[i][j] << " ";
            }
        std::cout << std::endl;
        }


    }
    MPI_Finalize();
    return 0;

}