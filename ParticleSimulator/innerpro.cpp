#include<mpi.h>
#include<vector> 
#include<iostream>
#define N 1000000

template<typename T>
void fillVector(std::vector<T> &v) {
    v.resize(N);
    for (int i = 0; i < N; i++) {
        v[i] = rand()%100;;
    }
}

template<typename T>
void vectorSum(std::vector<T>&v, std::vector<T> &gv, MPI_Comm &comm) {

    // MPI_Allreduce(v.data(), gv.data(), N, MPI_INT, MPI_SUM, comm);
    MPI_Reduce(&v[0], &gv[0], N, MPI_INT, MPI_SUM, 0, comm);
}




int main (int argc, char **argv) {

    int rank, size;
    int initialize, finalize;
    MPI_Initialized(&initialize);

    if (initialize == 0) {
        MPI_Init(&argc, &argv);
    }
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::vector<int> v;
    std::vector<int> gv;
    fillVector<int>(v);
    gv.resize(N);
    vectorSum<int>(v, gv, comm);
    if (rank == 0) {
        for (int i = 0; i < 20; i++) {
            std::cout << gv[i] << " ";
        }
        std::cout << std::endl;
    }
    MPI_Finalize();

}