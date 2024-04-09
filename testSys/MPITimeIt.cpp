#include <iostream>
#include<ctime>
#include <mpi.h>
#include <unistd.h> // for usleep()

int main(int argc, char** argv) {

    int mype;
    // Initialize MPI and get process rank
    
    MPI_Init(&argc, &argv);
    // Gets the rank of the calling process within the communicator MPI_COMM_WORLD.
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    
    // Start timer
    
    double t1 {MPI_Wtime()};
    sleep(10);
    double t2 {MPI_Wtime()};
    if (mype == 0) {
        printf( "Elapsed time is %f secs\n", t2 - t1 );
    }
    // Shutdown MPI
   MPI_Finalize();
//    return 0;
}