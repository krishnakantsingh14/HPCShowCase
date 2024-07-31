//https://github.com/lmontigny/MPI/blob/master/ETH_course/topology1.cpp
// https://www.youtube.com/watch?v=dc7bevHHplA
#include<mpi.h>
#include<iostream>


int main(int argc, char **argv) {

    int rank, size;
    MPI_Comm cart_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int ndim = 2;
    int dims[2]  = {0,0}; 
    int periods[2] = {0,0};
    int reorder = 1;
    int coords[2];
    int toporank;
    MPI_Comm carttopo;

    MPI_Dims_create(size, 3, dims);
    if (rank==0)
        std::cout << "We create a " << dims[0] << "x" << dims[1] << " arrangement.\n";

    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, reorder, &carttopo);
    MPI_Comm_rank(carttopo, &toporank); 
    int left, right, up, down;

    MPI_Cart_shift(carttopo,0, 1, &left, &right);
    MPI_Cart_shift(carttopo, 1, 1, &up, &down);
    MPI_Cart_coords(carttopo, toporank, 2, coords);


    int DATASEND = rank*50;
    int DATARECEIVE[4];
    MPI_Request request[8];
    MPI_Isend(&DATASEND, 1, MPI_INT, left,0, carttopo, &request[0]);
    MPI_Isend(&DATARECEIVE, 1, MPI_INT, right, 0, carttopo, &request[1]);
    MPI_Isend(&DATARECEIVE, 1, MPI_INT, up, 0, carttopo, &request[2]);
    MPI_Isend(&DATARECEIVE, 1, MPI_INT, down, 0, carttopo, &request[3]);

    MPI_Irecv(&DATARECEIVE[0], 1, MPI_INT, left, 0, carttopo, &request[4]);
    MPI_Irecv(&DATARECEIVE[1], 1, MPI_INT, right, 0, carttopo, &request[5]);
    MPI_Irecv(&DATARECEIVE[2], 1, MPI_INT, up, 0, carttopo, &request[6]);
    MPI_Irecv(&DATARECEIVE[3], 1, MPI_INT, down, 0, carttopo, &request[7]);

    MPI_Waitall(8, request, MPI_STATUS_IGNORE);

    
    for (int i = 0; i < 4; i++) {
        std::cout << "Rank" << rank << " " << DATARECEIVE[i] << std::endl;
    }



    // std::cout << "Rank " << rank << " " << toporank << " " << coords[0] << " " << coords[1] << " " << "Neighbours" <<
    // " " << left << " " << right << " " << up << " " << down << std::endl;
    MPI_Comm_free(&carttopo);
    MPI_Finalize();

}