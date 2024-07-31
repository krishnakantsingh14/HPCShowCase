#include"atom.h"
#include<mpi.h>
#include<vector>

int main (int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm comm = MPI_COMM_WORLD;

    

    MPI_Request request;
    MPI_Status status;

    // double *buffer = new double[buffersize];
    if (rank==0) {
        int NGHOST = 3;
        int buffersize = NGHOST* ATOM::atom<double>::bufferSize(comm);

        void *buffer = malloc(buffersize);

        std::vector<ATOM::atom<double>> ghostAtoms;

        for (int i=0; i<NGHOST; i++) {
            ghostAtoms.push_back(ATOM::atom<double>(rand()%100, rand()%100, rand()%100));
        }

        int position =0 ;
        for (auto ele: ghostAtoms) {
            ele.pack(comm, buffer, buffersize, &position);
        }
        std::cout << "Process 0 sent atoms data" << std::endl;

        MPI_Send(buffer, position, MPI_PACKED, 1, 0, comm);
    }

    else if (rank==1) {

        MPI_Probe(0, 0, comm, &status);
        int bufferSize;

        MPI_Get_count(&status, MPI_PACKED, &bufferSize);

        void *buffer = malloc(bufferSize);

        std::cout << "Process 1 receiving atoms data " << bufferSize << std::endl;
        MPI_Irecv(buffer, bufferSize, MPI_PACKED, 0, 0, comm, &request);
        MPI_Wait(&request, &status);
        // MPI_Wait();
        std::cout << "Process 1 received atoms data" << std::endl;
        int NGHOST = bufferSize/ATOM::atom<double>::bufferSize(comm);

        std::vector<ATOM::atom<double>> atoms(NGHOST);

        int position = 0;
        for (auto & ele: atoms) {
            ele.unpack(comm, buffer, bufferSize, &position);
            ele.printPosition();
            // cout<<
    }

    }

    MPI_Finalize();
    return 0;
}