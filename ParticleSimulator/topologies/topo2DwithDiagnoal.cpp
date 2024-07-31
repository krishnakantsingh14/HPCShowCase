#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int dims[2], periods[2] = {0, 0};
    int coords[2];
    MPI_Comm cart_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Define the 2D grid dimensions
    dims[0] = dims[1] = 0;
    MPI_Dims_create(size, 2, dims);

    // Create the Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Get the rank of neighbors in each direction
    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    // Initialize diagonal neighbors
    int north_west, north_east, south_west, south_east;
    north_west = north_east = south_west = south_east = MPI_PROC_NULL;

    // Determine the coordinates of diagonal neighbors
    if (north != MPI_PROC_NULL && west != MPI_PROC_NULL) {
        MPI_Cart_coords(cart_comm, north, 2, coords);
        coords[1]--;
        MPI_Cart_rank(cart_comm, coords, &north_west);
    }

    if (north != MPI_PROC_NULL && east != MPI_PROC_NULL) {
        MPI_Cart_coords(cart_comm, north, 2, coords);
        coords[1]++;
        MPI_Cart_rank(cart_comm, coords, &north_east);
    }

    if (south != MPI_PROC_NULL && west != MPI_PROC_NULL) {
        MPI_Cart_coords(cart_comm, south, 2, coords);
        coords[1]--;
        MPI_Cart_rank(cart_comm, coords, &south_west);
    }

    if (south != MPI_PROC_NULL && east != MPI_PROC_NULL) {
        MPI_Cart_coords(cart_comm, south, 2, coords);
        coords[1]++;
        MPI_Cart_rank(cart_comm, coords, &south_east);
    }

    printf("Rank %d at coords (%d, %d):\n", rank, coords[0], coords[1]);
    if (north_west != MPI_PROC_NULL) printf("  North-west neighbor rank: %d\n", north_west);
    if (north_east != MPI_PROC_NULL) printf("  North-east neighbor rank: %d\n", north_east);
    if (south_west != MPI_PROC_NULL) printf("  South-west neighbor rank: %d\n", south_west);
    if (south_east != MPI_PROC_NULL) printf("  South-east neighbor rank: %d\n", south_east);

    MPI_Finalize();
    return 0;
}
