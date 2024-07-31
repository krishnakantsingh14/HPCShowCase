//https://github.com/lmontigny/MPI/blob/master/ETH_course/topology1.cpp
#include<mpi.h>
#include<iostream>


int main(int argc, char **argv) {

    int rank, size;
    MPI_Comm cart_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nums[3] = {0, 0, 0};
    int periodic[3] = {0, 0, 0};
    int coords[3];
    

    //split the nodes automatically
    // https://github.com/mpi-forum/mpi-forum-historic/issues/194 
    MPI_Dims_create(size, 3, nums); // -> This will fill up num
    
    if (rank==0) {
        std::cout << "We create a " << 
        nums[0] << "x" << 
        nums[1] << "x" << 
        nums[2] << " arrangement" << std::endl;
    }
    
    MPI_Cart_create(MPI_COMM_WORLD, 3, nums, periodic, 1, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    // std::cout<< rank << ":" << coords[0] << coords[1] << coords[2] << std::endl;

    int left, right, up, down, front, back;
    int toporank;
    
    MPI_Comm_rank(cart_comm, &toporank);


    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 2, 1, &front, &back);


    std::cout << "Rank " << rank << " has new rank " << toporank << " and neighbors "
            << left << ", " << right << ", " << up << ", " << down << ", "
            << front << ", " << back << std::endl;
  
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();

}